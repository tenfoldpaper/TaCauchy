"""
Simple Gelpad Test - No Motion
只导入模型，输出调试信息，不设置任何运动
"""

from __future__ import annotations

import argparse
import numpy as np

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Simple Gelpad Test - No Motion")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import carb

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from tacex_assets import TACEX_ASSETS_DATA_DIR
from tacex_assets.robots.franka.franka_gsmini_single_uipc import (
    FRANKA_PANDA_ARM_SINGLE_GSMINI_HIGH_PD_UIPC_CFG,
)

from tacex_uipc import (
    UipcIsaacAttachments,
    UipcIsaacAttachmentsCfg,
    UipcObject,
    UipcObjectCfg,
    UipcRLEnv,
    UipcSimCfg,
)


@configclass
class SimpleTestEnvCfg(DirectRLEnvCfg):
    """Simple test environment configuration."""
    
    viewer: ViewerCfg = ViewerCfg()
    decimation: int = 1
    
    # Episode length
    episode_length_s: float = 10.0
    
    # Dummy observation and action spaces (required by DirectRLEnvCfg)
    observation_space: int = 1
    action_space: int = 1
    
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=1,
    )

    uipc_sim = UipcSimCfg(
        ground_height=0.0025,
        contact=UipcSimCfg.Contact(d_hat=0.0001),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=1.5,
        replicate_physics=True,
    )

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, 0)),
        spawn=sim_utils.GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    robot: ArticulationCfg = FRANKA_PANDA_ARM_SINGLE_GSMINI_HIGH_PD_UIPC_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
    )
    
    # Gelpad 使用预计算的 tet mesh
    gelpad_cfg = UipcObjectCfg(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_gelpad",
        mesh_cfg=None,  # 使用预计算的 tet mesh
        constitution_cfg=UipcObjectCfg.StableNeoHookeanCfg(),
    )
    
    # Attachment 使用预计算数据
    gelpad_attachment_cfg = UipcIsaacAttachmentsCfg(
        constraint_strength_ratio=100.0, 
        body_name="gelsight_mini_case", 
        debug_vis=False,
        compute_attachment_data=False  # 使用预计算的 attachment 数据
    )


class SimpleTestEnv(UipcRLEnv):
    cfg: SimpleTestEnvCfg

    def __init__(self, cfg: SimpleTestEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.step_count = 0

    def _setup_scene(self):
        print("\n" + "=" * 70)
        print("SETTING UP SCENE")
        print("=" * 70)
        
        # Create robot
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        print(f"✓ Robot created: {self.cfg.robot.prim_path}")
        
        # Clone environments
        self.scene.clone_environments(copy_from_source=False)
        print(f"✓ Environments cloned: {self.num_envs}")

        # Create ground
        ground = self.cfg.ground
        ground.spawn.func(
            ground.prim_path, ground.spawn, 
            translation=ground.init_state.pos, 
            orientation=ground.init_state.rot
        )
        print(f"✓ Ground created")

        # Create light
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        print(f"✓ Light created")

        # --- UIPC simulation setup ---
        print("\n" + "-" * 70)
        print("UIPC SETUP")
        print("-" * 70)
        
        self._uipc_gelpad = UipcObject(self.cfg.gelpad_cfg, self.uipc_sim)
        print(f"✓ Gelpad created: {self.cfg.gelpad_cfg.prim_path}")

        # Create attachment
        print(f"\nCreating attachment:")
        print(f"  Gelpad prim: {self.cfg.gelpad_cfg.prim_path}")
        print(f"  Robot prim: {self.scene.articulations['robot'].cfg.prim_path}")
        print(f"  Body name: {self.cfg.gelpad_attachment_cfg.body_name}")
        
        self.attachment = UipcIsaacAttachments(
            self.cfg.gelpad_attachment_cfg, 
            self._uipc_gelpad, 
            self.scene.articulations["robot"]
        )
        print(f"✓ Attachment created with {len(self.attachment.attachment_points_idx)} attachment points")
        print("=" * 70 + "\n")

    def _pre_physics_step(self, actions: torch.Tensor):
        pass

    def _apply_action(self):
        # 不设置任何动作，让机器人保持初始姿态
        pass

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device), \
               torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        # 保持初始关节位置
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    def _get_observations(self) -> dict:
        return {}


def print_debug_info(env: SimpleTestEnv):
    """打印调试信息"""
    print("\n" + "=" * 70)
    print("DEBUG INFORMATION")
    print("=" * 70)
    
    # 1. Robot information
    print("\n1. ROBOT INFORMATION")
    print("-" * 70)
    print(f"Robot prim path: {env._robot.cfg.prim_path}")
    print(f"Number of bodies: {env._robot.num_bodies}")
    print(f"Body names: {env._robot.body_names}")
    
    # Get robot root position
    root_pos = env._robot.data.root_pos_w.cpu().numpy()
    root_quat = env._robot.data.root_quat_w.cpu().numpy()
    print(f"\nRobot root position: [{root_pos[0, 0]:.4f}, {root_pos[0, 1]:.4f}, {root_pos[0, 2]:.4f}]")
    print(f"Robot root quaternion: [{root_quat[0, 0]:.4f}, {root_quat[0, 1]:.4f}, {root_quat[0, 2]:.4f}, {root_quat[0, 3]:.4f}]")
    
    # 2. Case information
    print("\n2. CASE INFORMATION")
    print("-" * 70)
    case_body_idx = env._robot.find_bodies("gelsight_mini_case")[0][0]
    case_pos = env._robot.data.body_link_pos_w[:, case_body_idx].cpu().numpy()
    case_quat = env._robot.data.body_link_quat_w[:, case_body_idx].cpu().numpy()
    print(f"Case body index: {case_body_idx}")
    print(f"Case position: [{case_pos[0, 0]:.4f}, {case_pos[0, 1]:.4f}, {case_pos[0, 2]:.4f}]")
    print(f"Case quaternion: [{case_quat[0, 0]:.4f}, {case_quat[0, 1]:.4f}, {case_quat[0, 2]:.4f}, {case_quat[0, 3]:.4f}]")
    
    # 3. Gelpad information
    print("\n3. GELPAD INFORMATION")
    print("-" * 70)
    print(f"Gelpad prim path: {env._uipc_gelpad.cfg.prim_path}")
    print(f"Gelpad object ID: {env._uipc_gelpad.obj_id}")
    
    # Force update the gelpad data
    env._uipc_gelpad.update(0.0)
    
    # Get gelpad vertex positions from data (cached)
    if hasattr(env._uipc_gelpad, 'data') and hasattr(env._uipc_gelpad.data, 'nodal_pos_w'):
        gelpad_pos = env._uipc_gelpad.data.nodal_pos_w.cpu().numpy()
        gelpad_center = np.mean(gelpad_pos, axis=0)
        print(f"Gelpad vertices (from data): {gelpad_pos.shape[0]}")
        print(f"Gelpad center (from data): [{gelpad_center[0]:.4f}, {gelpad_center[1]:.4f}, {gelpad_center[2]:.4f}]")
        print(f"Gelpad X range: [{gelpad_pos[:, 0].min():.4f}, {gelpad_pos[:, 0].max():.4f}]")
        print(f"Gelpad Y range: [{gelpad_pos[:, 1].min():.4f}, {gelpad_pos[:, 1].max():.4f}]")
        print(f"Gelpad Z range: [{gelpad_pos[:, 2].min():.4f}, {gelpad_pos[:, 2].max():.4f}]")
    
    # Get gelpad vertex positions directly from UIPC (no cache)
    geom = env.uipc_sim.scene.geometries()
    geo_slot, geo_slot_rest = geom.find(env._uipc_gelpad.obj_id)
    direct_pos = geo_slot.geometry().positions().view().reshape(-1, 3)
    direct_center = np.mean(direct_pos, axis=0)
    print(f"\nGelpad center (direct from UIPC): [{direct_center[0]:.4f}, {direct_center[1]:.4f}, {direct_center[2]:.4f}]")
    print(f"Direct Z range: [{direct_pos[:, 2].min():.4f}, {direct_pos[:, 2].max():.4f}]")
    
    # Get initial vertex positions
    if hasattr(env._uipc_gelpad, 'init_vertex_pos'):
        init_pos = env._uipc_gelpad.init_vertex_pos.cpu().numpy()
        init_center = np.mean(init_pos, axis=0)
        print(f"\nInitial gelpad center: [{init_center[0]:.4f}, {init_center[1]:.4f}, {init_center[2]:.4f}]")
        print(f"Initial Z range: [{init_pos[:, 2].min():.4f}, {init_pos[:, 2].max():.4f}]")
    
    # 4. Attachment information
    print("\n4. ATTACHMENT INFORMATION")
    print("-" * 70)
    print(f"Number of attachment points: {len(env.attachment.attachment_points_idx)}")
    print(f"Attachment point indices: {env.attachment.attachment_points_idx}")
    
    if hasattr(env.attachment, 'aim_positions') and env.attachment.aim_positions.shape[0] > 0:
        aim_pos = env.attachment.aim_positions
        aim_center = np.mean(aim_pos, axis=0)
        print(f"\nAttachment aim positions shape: {aim_pos.shape}")
        print(f"Attachment aim center: [{aim_center[0]:.4f}, {aim_center[1]:.4f}, {aim_center[2]:.4f}]")
        print(f"Attachment aim Z range: [{aim_pos[:, 2].min():.4f}, {aim_pos[:, 2].max():.4f}]")
    
    # 5. Relative positions
    print("\n5. RELATIVE POSITIONS")
    print("-" * 70)
    if hasattr(env._uipc_gelpad, 'data') and hasattr(env._uipc_gelpad.data, 'nodal_pos_w'):
        gelpad_pos = env._uipc_gelpad.data.nodal_pos_w.cpu().numpy()
        gelpad_center = np.mean(gelpad_pos, axis=0)
        
        # Distance between case and gelpad
        distance = np.linalg.norm(case_pos[0] - gelpad_center)
        print(f"Distance between case and gelpad center: {distance:.4f} m")
        
        # Z difference
        z_diff = case_pos[0, 2] - gelpad_center[2]
        print(f"Z difference (case - gelpad): {z_diff:.4f} m")
        
        if abs(z_diff) > 0.1:
            print(f"⚠️  WARNING: Large Z difference! Gelpad should be close to case.")
        else:
            print(f"✓ Z difference is reasonable")
    
    print("\n" + "=" * 70 + "\n")


def run_simulator(env: SimpleTestEnv):
    """运行模拟器，不设置任何运动"""
    
    print("\n" + "=" * 70)
    print("STARTING SIMULATION")
    print("=" * 70)
    print("No motion will be applied - robot stays in initial pose")
    print("Press Ctrl+C to stop")
    print("=" * 70 + "\n")
    
    env.reset()
    
    # 打印初始调试信息
    print_debug_info(env)
    
    step_count = 0
    print_interval = 60  # 每60步打印一次
    
    try:
        while simulation_app.is_running():
            # 不应用任何动作
            env.scene.write_data_to_sim()
            env.sim.step(render=True)
            
            # 更新 UIPC 渲染网格
            env.uipc_sim.update_render_meshes()
            
            # 更新场景
            env.scene.update(dt=env.physics_dt)
            
            step_count += 1
            
            # 定期打印调试信息
            if step_count % print_interval == 0:
                print(f"\n[Step {step_count}]")
                print_debug_info(env)
            
    except KeyboardInterrupt:
        print("\n\nSimulation stopped by user")
    finally:
        env.close()


def main():
    """Main function."""
    env_cfg = SimpleTestEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    env = SimpleTestEnv(env_cfg)

    print("[INFO]: Setup complete...")
    run_simulator(env=env)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        carb.log_error(str(err))
        import traceback
        carb.log_error(traceback.format_exc())
        raise
    finally:
        simulation_app.close()
