"""
Environment configuration for simple tactile demo.

This module provides a clean interface to create tactile sensor environments
with multiple UIPC objects for interaction.
"""

import numpy as np
import torch
from isaaclab.utils import configclass
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, ArticulationCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaacsim.core.api.objects import VisualCuboid

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg

from tacex import GelSightSensor
from tacex_assets import TACEX_ASSETS_DATA_DIR
from tacex_assets.robots.franka.franka_gsmini_single_uipc import FRANKA_PANDA_ARM_SINGLE_GSMINI_HIGH_PD_UIPC_CFG
from tacex_assets.robots.franka.franka_digit_single_uipc import FRANKA_PANDA_ARM_SINGLE_DIGIT_HIGH_PD_UIPC_CFG
from tacex_assets.robots.franka.franka_9dtact_single_uipc import FRANKA_PANDA_ARM_SINGLE_9DTACT_HIGH_PD_UIPC_CFG
from tacex_assets.sensors.gelsight_mini.gsmini_cfg import GelSightMiniCfg
from tacex_assets.sensors.ninedtact import NineDTactCfg

from tacex_uipc import (
    UipcObject, UipcObjectCfg,
    UipcIsaacAttachments, UipcIsaacAttachmentsCfg,
    UipcRLEnv, UipcSimCfg,
)
from isaaclab.assets import Articulation, RigidObject



@configclass
class SimpleTactileEnvCfg(DirectRLEnvCfg):
    """Configuration for simple tactile environment."""
    
    viewer: ViewerCfg = ViewerCfg()
    viewer.eye = (1.9, 1.4, 0.3)
    viewer.lookat = (-1.5, -1.9, -1.1)
    
    debug_vis = False
    decimation = 1
    sensor_type: str = "gelsight"
    
    sim: SimulationCfg = SimulationCfg(
        dt=1/60,
        render_interval=decimation,
        physx=PhysxCfg(enable_ccd=True),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=5.0,
            dynamic_friction=5.0,
            restitution=0.0,
        ),
    )
    
    uipc_sim = UipcSimCfg(
        ground_height=0.0025,
        contact=UipcSimCfg.Contact(d_hat=0.0001),
        gravity=(0.0, 0.0, -9.8),
        dt=1/60,
    )
    
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=1.5,
        replicate_physics=True,
        lazy_sensor_update=True,
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
    
    # UIPC objects for tactile interaction
    # Only sphere2 at center position
    uipc_shapes = {
        "sphere2": UipcObjectCfg(
            prim_path="/World/envs/env_.*/sphere2_uipc",
            init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0.0, 0.03]),  # Centered
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{TACEX_ASSETS_DATA_DIR}/Props/tactile_test_shapes/sphere2.usd",
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            ),
            mesh_cfg=None,
            constitution_cfg=UipcObjectCfg.AffineBodyConstitutionCfg(m_kappa=100.0, kinematic=True),
            mass_density=1e3,
        ),
    }
    
    # PhysX rigid body objects (视觉参考，不参与 UIPC 交互)
    # Only sphere2 at center position
    physx_shapes = {
        "sphere2_physx": RigidObjectCfg(
            prim_path="/World/envs/env_.*/sphere2_physx",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0.0, 0.03]),  # Centered
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{TACEX_ASSETS_DATA_DIR}/Props/tactile_test_shapes/sphere2.usd",
                rigid_props=RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=True,
                ),
            ),
        ),
    }
    
    ik_controller_cfg = DifferentialIKControllerCfg(
        command_type="pose", use_relative_mode=False, ik_method="dls"
    )
    
    # Dynamic configs (set by create_tactile_env)
    robot: ArticulationCfg = None
    gelpad_cfg: UipcObjectCfg = None
    gelpad_attachment_cfg: UipcIsaacAttachmentsCfg = None
    gsmini: GelSightMiniCfg = None
    
    episode_length_s = 0
    action_space = 0
    observation_space = 0
    state_space = 0



class SimpleTactileEnv(UipcRLEnv):
    """Simple tactile environment with goal-based control."""
    
    cfg: SimpleTactileEnvCfg
    
    def __init__(self, cfg: SimpleTactileEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # IK controller
        self._ik_controller = DifferentialIKController(
            cfg=self.cfg.ik_controller_cfg, num_envs=self.num_envs, device=self.device
        )
        
        body_ids, body_names = self._robot.find_bodies("panda_hand")
        self._body_idx = body_ids[0]
        self._jacobi_body_idx = self._body_idx - 1
        
        # Sensor-specific offset
        if self.cfg.sensor_type == "digit":
            self._offset_pos = torch.tensor([0.0, 0.0, 0.132], device=self.device).repeat(self.num_envs, 1)
        elif self.cfg.sensor_type == "9dtact":
            self._offset_pos = torch.tensor([0.0, 0.0, 0.131], device=self.device).repeat(self.num_envs, 1)
        else:
            self._offset_pos = torch.tensor([0.0, 0.0, 0.131], device=self.device).repeat(self.num_envs, 1)
        
        self._offset_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        self.ik_commands = torch.zeros((self.num_envs, self._ik_controller.action_dim), device=self.device)
        self.goal_prim_view = None
    
    def _setup_scene(self):
        # Robot
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self.scene.clone_environments(copy_from_source=False)
        
        # End-effector frame
        ee_offset_z = 0.132 if self.cfg.sensor_type == "digit" else 0.131
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        ee_frame_cfg = FrameTransformerCfg(
            prim_path="/World/envs/env_.*/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="/World/envs/env_.*/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(pos=(0.0, 0.0, ee_offset_z)),
                ),
            ],
        )
        self._ee_frame = FrameTransformer(ee_frame_cfg)
        self.scene.sensors["ee_frame"] = self._ee_frame
        
        # Sensor
        self.gsmini = GelSightSensor(self.cfg.gsmini)
        self.scene.sensors["gsmini"] = self.gsmini
        
        # Ground
        self.cfg.ground.spawn.func(
            self.cfg.ground.prim_path, self.cfg.ground.spawn,
            translation=self.cfg.ground.init_state.pos,
            orientation=self.cfg.ground.init_state.rot
        )
        
        # Goal marker
        VisualCuboid(
            prim_path="/World/envs/env_0/goal",
            size=0.01,
            position=np.array([0.5, 0.0, 0.027]),
            orientation=np.array([0, 1, 0, 0]),
            color=np.array([255.0, 0.0, 0.0]),
            visible=True,
        )
        
        # Light
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        
        # UIPC gelpad
        self._uipc_gelpad = UipcObject(self.cfg.gelpad_cfg, self.uipc_sim)
        
        # UIPC shapes
        self.uipc_shapes = {}
        for shape_name, shape_cfg in self.cfg.uipc_shapes.items():
            self.uipc_shapes[shape_name] = UipcObject(shape_cfg, self.uipc_sim)
            self.scene.uipc_objects[f"{shape_name}_uipc"] = self.uipc_shapes[shape_name]
        
        # PhysX shapes (视觉参考，不参与 UIPC 交互)
        self.physx_shapes = {}
        for shape_name, shape_cfg in self.cfg.physx_shapes.items():
            self.physx_shapes[shape_name] = RigidObject(shape_cfg)
            self.scene.rigid_objects[shape_name] = self.physx_shapes[shape_name]
        
        # Attachment
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
    
    def _get_observations(self) -> dict:
        return {}
    
    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (torch.zeros(self.num_envs, device=self.device, dtype=torch.bool),
                torch.zeros(self.num_envs, device=self.device, dtype=torch.bool))
    
    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    
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
    
    def _compute_frame_pose(self):
        ee_pos_w = self._robot.data.body_link_pos_w[:, self._body_idx]
        ee_quat_w = self._robot.data.body_link_quat_w[:, self._body_idx]
        root_pos_w = self._robot.data.root_link_pos_w
        root_quat_w = self._robot.data.root_link_quat_w
        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
        ee_pos_b, ee_quat_b = math_utils.combine_frame_transforms(
            ee_pos_b, ee_quat_b, self._offset_pos, self._offset_rot
        )
        return ee_pos_b, ee_quat_b
    
    def _compute_frame_jacobian(self):
        jacobian = self.jacobian_b
        jacobian[:, 0:3, :] += torch.bmm(-math_utils.skew_symmetric_matrix(self._offset_pos), jacobian[:, 3:, :])
        jacobian[:, 3:, :] = torch.bmm(math_utils.matrix_from_quat(self._offset_rot), jacobian[:, 3:, :])
        return jacobian



def create_tactile_env(sensor_type: str = "gelsight", num_envs: int = 1, debug_vis: bool = False):
    """
    Create a tactile sensor environment with specified configuration.
    
    Args:
        sensor_type: "gelsight", "digit", or "9dtact"
        num_envs: Number of parallel environments
        debug_vis: Enable debug visualization
        
    Returns:
        Configured SimpleTactileEnv instance
    """
    # Create base config
    cfg = SimpleTactileEnvCfg()
    cfg.sensor_type = sensor_type
    cfg.scene.num_envs = num_envs
    cfg.debug_vis = debug_vis
    
    # Robot configuration (use replace to properly copy all fields)
    if sensor_type == "digit":
        cfg.robot = FRANKA_PANDA_ARM_SINGLE_DIGIT_HIGH_PD_UIPC_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    elif sensor_type == "9dtact":
        cfg.robot = FRANKA_PANDA_ARM_SINGLE_9DTACT_HIGH_PD_UIPC_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    else:
        cfg.robot = FRANKA_PANDA_ARM_SINGLE_GSMINI_HIGH_PD_UIPC_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    
    # Sensor configuration
    tactile_res = (640, 480)
    camera_res = (320, 240)
    
    if sensor_type == "digit":
        cfg.gsmini = GelSightMiniCfg(
            prim_path="/World/envs/env_.*/Robot/digit_case",
            sensor_camera_cfg=GelSightMiniCfg.SensorCameraCfg(
                prim_path_appendix="/Camera",
                update_period=0,
                resolution=camera_res,
                data_types=["depth"],
                clipping_range=(0.024, 0.034),
            ),
            device="cuda",
            debug_vis=debug_vis,
            marker_motion_sim_cfg=None,
            data_types=["tactile_rgb"],
        )
        cfg.gsmini.optical_sim_cfg = cfg.gsmini.optical_sim_cfg.replace(
            calib_folder_path=f"{TACEX_ASSETS_DATA_DIR}/Sensors/digit/calibs/640x480",
            with_shadow=False,
            device="cuda",
            tactile_img_res=tactile_res,
        )
    elif sensor_type == "9dtact":
        cfg.gsmini = NineDTactCfg(
            prim_path="/World/envs/env_.*/Robot/case_9dtact",
            sensor_camera_cfg=NineDTactCfg.SensorCameraCfg(
                prim_path_appendix="/Camera",
                update_period=0,
                resolution=camera_res,
                data_types=["depth"],
                clipping_range=(0.024, 0.034),
            ),
            device="cuda",
            debug_vis=debug_vis,
            marker_motion_sim_cfg=None,
            data_types=["tactile_rgb"],
        )
        cfg.gsmini.optical_sim_cfg = cfg.gsmini.optical_sim_cfg.replace(
            calib_folder_path=f"{TACEX_ASSETS_DATA_DIR}/Sensors/9dtact/calibs/640x480",
            with_shadow=False,
            device="cuda",
            tactile_img_res=tactile_res,
            convert_to_grayscale=True,
            apply_color_correction=True,
        )
    else:  # gelsight
        cfg.gsmini = GelSightMiniCfg(
            prim_path="/World/envs/env_.*/Robot/gelsight_mini_case",
            sensor_camera_cfg=GelSightMiniCfg.SensorCameraCfg(
                prim_path_appendix="/Camera",
                update_period=0,
                resolution=camera_res,
                data_types=["depth"],
                clipping_range=(0.024, 0.034),
            ),
            device="cuda",
            debug_vis=debug_vis,
            marker_motion_sim_cfg=None,
            data_types=["tactile_rgb"],
        )
        cfg.gsmini.optical_sim_cfg = cfg.gsmini.optical_sim_cfg.replace(
            with_shadow=False,
            device="cuda",
            tactile_img_res=tactile_res,
        )
    
    # Gelpad configuration
    youngs_modulus = 10.0
    if sensor_type == "digit":
        cfg.gelpad_cfg = UipcObjectCfg(
            prim_path="/World/envs/env_.*/Robot/digit_gelpad",
            mesh_cfg=None,
            constitution_cfg=UipcObjectCfg.StableNeoHookeanCfg(
                youngs_modulus=youngs_modulus, poisson_rate=0.40
            ),
        )
    elif sensor_type == "9dtact":
        cfg.gelpad_cfg = UipcObjectCfg(
            prim_path="/World/envs/env_.*/Robot/Gelpad9DTact",
            mesh_prim_name="mesh",
            mesh_cfg=None,
            constitution_cfg=UipcObjectCfg.StableNeoHookeanCfg(
                youngs_modulus=youngs_modulus, poisson_rate=0.40
            ),
        )
    else:
        cfg.gelpad_cfg = UipcObjectCfg(
            prim_path="/World/envs/env_.*/Robot/gelsight_mini_gelpad",
            mesh_cfg=None,
            constitution_cfg=UipcObjectCfg.StableNeoHookeanCfg(
                youngs_modulus=youngs_modulus, poisson_rate=0.40
            ),
        )
    
    # Attachment configuration
    if sensor_type == "digit":
        body_name = "digit_case"
    elif sensor_type == "9dtact":
        body_name = "case_9dtact"
    else:
        body_name = "gelsight_mini_case"
    
    cfg.gelpad_attachment_cfg = UipcIsaacAttachmentsCfg(
        constraint_strength_ratio=2000.0,
        body_name=body_name,
        debug_vis=False,
        compute_attachment_data=True,
        constrain_entire_surface=False,
    )
    
    # Create environment
    env = SimpleTactileEnv(cfg)
    return env
