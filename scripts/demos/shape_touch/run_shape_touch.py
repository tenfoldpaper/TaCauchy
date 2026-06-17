"""Collecting tactile data of the shapes from https://danfergo.github.io/gelsight-simulation/.
#720行控制广角参数

Use
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Control Franka, which is equipped with one tactile sensor (GelSight Mini or DIGIT), by moving the Frame in the GUI"
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--sys", type=bool, default=True, help="Whether to track system utilization.")
parser.add_argument(
    "--debug_vis",
    default=True,
    action="store_true",
    help="Whether to render tactile images in the GUI",
)
parser.add_argument(
    "--sensor_type",
    type=str,
    default="gsmini",
    choices=["gsmini", "digit"],
    help="Type of tactile sensor to use: 'gsmini' for GelSight Mini or 'digit' for DIGIT sensor",
)
AppLauncher.add_app_launcher_args(parser)
# parse the arguments

args_cli = parser.parse_args()
args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import traceback
from contextlib import suppress
from pathlib import Path

import omni.ui

with suppress(ImportError):
    # isaacsim.gui is not available when running in headless mode.
    import isaacsim.gui.components.ui_utils as ui_utils

import numpy as np
import torch
import cv2
import os
from datetime import datetime

import carb
import pynvml
from isaacsim.core.api.objects import VisualCuboid
from isaacsim.core.prims import XFormPrim

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import (
    Articulation,
    ArticulationCfg,
    AssetBaseCfg,
    RigidObject,
    RigidObjectCfg,
)
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim import PhysxCfg, RenderCfg, SimulationCfg
from isaaclab.utils import configclass

import omni.usd

from tacex import GelSightSensor
from tacex.simulation_approaches.fots import FOTSMarkerSimulatorCfg

from tacex_assets import TACEX_ASSETS_DATA_DIR
from tacex_assets.robots.franka.franka_gsmini_single_rigid import (
    FRANKA_PANDA_ARM_SINGLE_GSMINI_HIGH_PD_RIGID_CFG,
)
from tacex_assets.robots.franka.franka_digit_single_rigid import (
    FRANKA_PANDA_ARM_SINGLE_DIGIT_HIGH_PD_RIGID_CFG,
)
from tacex_assets.sensors.gelsight_mini.gsmini_cfg import GelSightMiniCfg
from tacex_assets.sensors.digit.digit_cfg import DigitCfg

#  from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
# from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg

# Global variables for image saving
SAVE_DIR = "/home/ly/data/hengfei/getdata/digit/sim"
save_counter = 0
current_tactile_image = None


def mouse_callback(event, x, y, flags, param):
    """Mouse callback function for OpenCV window to save images on click."""
    global save_counter, current_tactile_image
    
    if event == cv2.EVENT_LBUTTONDOWN and current_tactile_image is not None:
        # Create save directory if it doesn't exist
        os.makedirs(SAVE_DIR, exist_ok=True)
        
        # Generate filename with timestamp and counter
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tactile_{timestamp}_{save_counter:04d}.png"
        filepath = os.path.join(SAVE_DIR, filename)
        
        # Save the image
        cv2.imwrite(filepath, current_tactile_image)
        save_counter += 1
        
        print(f"[INFO] Saved tactile image to: {filepath}")


class CustomEnvWindow(BaseEnvWindow):
    """Window manager for the RL environment."""

    def __init__(self, env: DirectRLEnvCfg, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)

        self.object_names = list(create_shapes_cfg().keys())
        self.current_object_name = self.object_names[0]

        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)

        with self.ui_window_elements["main_vstack"]:
            self._build_control_frame()
            # collapse some frames which we don't need
            self.ui_window_elements["debug_frame"].collapsed = True
            self.ui_window_elements["sim_frame"].collapsed = True

    def _build_control_frame(self):
        self.ui_window_elements["action_frame"] = omni.ui.CollapsableFrame(
            title="Shape Touching Demo Script",
            width=omni.ui.Fraction(1),
            height=0,
            collapsed=False,
            style=ui_utils.get_style(),
            horizontal_scrollbar_policy=omni.ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
            vertical_scrollbar_policy=omni.ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
        )
        with self.ui_window_elements["action_frame"]:
            self.ui_window_elements["action_vstack"] = omni.ui.VStack(spacing=5, height=50)
            with self.ui_window_elements["action_vstack"]:
                objects_dropdown_cfg = {
                    "label": "Objects",
                    "type": "dropdown",
                    "default_val": 0,
                    "items": self.object_names,
                    "tooltip": "Select an action for the gripper",
                    "on_clicked_fn": self._set_main_obj,
                }
                self.ui_window_elements["object_dropdown"] = ui_utils.dropdown_builder(**objects_dropdown_cfg)

                self.ui_window_elements["reset_button"] = ui_utils.btn_builder(
                    type="button",
                    text="Reset Env",
                    tooltip="Resets the environment, i.e. the objects are spawned back at their initial position.",
                    on_clicked_fn=self._reset_env,
                )

    def get_current_object_name(self):
        current_obj_idx = (
            self.ui_window_elements["object_dropdown"].get_item_value_model().get_value_as_int()
        )  # dropdown options are returned as numbers -> index in list
        self.current_object_name = self.object_names[current_obj_idx]
        return self.current_object_name

    def _set_main_obj(self, value):
        print("Set new main obj ", value)
        # get pose of old obj and of new current obj
        old_obj: RigidObject = self.env.scene.rigid_objects[self.current_object_name]

        obj_name = self.get_current_object_name()
        new_obj: RigidObject = self.env.scene.rigid_objects[obj_name]
        new_pose = new_obj.data.root_pose_w

        # swap position of old and new obj
        old_obj.write_root_pose_to_sim(new_pose)
        new_obj.write_root_pose_to_sim(self.env.main_pose)

    def _reset_env(self):
        self.env.reset()


def create_shapes_cfg() -> dict[str, RigidObjectCfg]:
    """Creates RigidObjectCfg's for each usd file in the `{TACEX_ASSETS_DATA_DIR}/Props/tactile_test_shapes/` directory.

    The objects are spawned on a line along the y axis.
    Returns:
        shapes: dict of names and corresponding RigidObjectCfg's
    """
    shapes = {}
    usd_files_path = list(Path(f"{TACEX_ASSETS_DATA_DIR}/Props/tactile_test_shapes/").glob("*.usd"))

    for i, file_path in enumerate(usd_files_path):
        file_name = file_path.stem

        shapes[file_name] = RigidObjectCfg(
            prim_path=f"/World/envs/env_.*/{file_name}",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0.025 * i, 0.02]),
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(file_path),
                # scale=(0.8, 0.8, 0.8),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    kinematic_enabled=True,
                    disable_gravity=False,
                ),
            ),
        )

    return shapes


def create_robot_cfg(sensor_type: str) -> ArticulationCfg:
    """Create robot configuration based on sensor type."""
    if sensor_type == "digit":
        # Use predefined DIGIT configuration
        robot_cfg = FRANKA_PANDA_ARM_SINGLE_DIGIT_HIGH_PD_RIGID_CFG.copy()
    else:  # gsmini
        robot_cfg = FRANKA_PANDA_ARM_SINGLE_GSMINI_HIGH_PD_RIGID_CFG.copy()
    
    robot_cfg = robot_cfg.replace(prim_path="/World/envs/env_.*/Robot")
    return robot_cfg


def create_sensor_cfg(sensor_type: str, shapes: dict):
    """Create sensor configuration based on sensor type."""
    # setup marker for FOTS frame_transformer
    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
    marker_cfg.prim_path = "/Visuals/FrameTransformer"

    if sensor_type == "digit":
        # DIGIT sensor configuration - aligned with GelSight Mini
        sensor_cfg = DigitCfg(
            prim_path="/World/envs/env_.*/Robot/digit_case",
            sensor_camera_cfg=DigitCfg.SensorCameraCfg(
                prim_path_appendix="/Camera",
                update_period=0,
                resolution=(320, 240),
                data_types=["depth"],
                clipping_range=(0.024, 0.034),
            ),
            device="cuda",
            debug_vis=False,
            marker_motion_sim_cfg=FOTSMarkerSimulatorCfg(
                lamb=[0.00125, 0.00021, 0.00038],
                pyramid_kernel_size=[51, 21, 11, 5],
                kernel_size=5,
                marker_params=FOTSMarkerSimulatorCfg.MarkerParams(
                    num_markers_col=11,
                    num_markers_row=9,
                    num_markers=99,
                    x0=15,
                    y0=26,
                    dx=26,
                    dy=29,
                ),
                tactile_img_res=(320, 240),
                device="cuda",
                frame_transformer_cfg=FrameTransformerCfg(
                    prim_path="/World/envs/env_.*/Robot/digit_gelpad",
                    source_frame_offset=OffsetCfg(),
                    target_frames=[
                        FrameTransformerCfg.FrameCfg(prim_path=f"/World/envs/env_.*/{obj_name}")
                        for obj_name in list(shapes.keys())
                    ],
                    debug_vis=True,
                    visualizer_cfg=marker_cfg,
                ),
            ),
            data_types=["marker_motion", "tactile_rgb"],
        )
        # Change settings for optical sim to match GelSight Mini
        sensor_cfg.optical_sim_cfg = sensor_cfg.optical_sim_cfg.replace(
            with_shadow=False,
            device="cuda",
            tactile_img_res=(320, 240),
        )
    else:  # gsmini
        sensor_cfg = GelSightMiniCfg(
            prim_path="/World/envs/env_.*/Robot/gelsight_mini_case",
            sensor_camera_cfg=GelSightMiniCfg.SensorCameraCfg(
                prim_path_appendix="/Camera",
                update_period=0,
                resolution=(320, 240),
                data_types=["depth"],
                clipping_range=(0.024, 0.034),
            ),
            device="cuda",
            debug_vis=True,
            marker_motion_sim_cfg=FOTSMarkerSimulatorCfg(
                lamb=[0.00125, 0.00021, 0.00038],
                pyramid_kernel_size=[51, 21, 11, 5],
                kernel_size=5,
                marker_params=FOTSMarkerSimulatorCfg.MarkerParams(
                    num_markers_col=11,
                    num_markers_row=9,
                    num_markers=99,
                    x0=15,
                    y0=26,
                    dx=26,
                    dy=29,
                ),
                tactile_img_res=(320, 240),
                device="cuda",
                frame_transformer_cfg=FrameTransformerCfg(
                    prim_path="/World/envs/env_.*/Robot/gelsight_mini_gelpad",
                    source_frame_offset=OffsetCfg(),
                    target_frames=[
                        FrameTransformerCfg.FrameCfg(prim_path=f"/World/envs/env_.*/{obj_name}")
                        for obj_name in list(shapes.keys())
                    ],
                    debug_vis=True,
                    visualizer_cfg=marker_cfg,
                ),
            ),
            data_types=["marker_motion", "tactile_rgb"],
        )
        # change settings for optical sim
        sensor_cfg.optical_sim_cfg = sensor_cfg.optical_sim_cfg.replace(
            with_shadow=False,
            device="cuda",
            tactile_img_res=(320, 240),
        )
    
    return sensor_cfg


@configclass
class BallRollingEnvCfg(DirectRLEnvCfg):
    # viewer settings
    viewer: ViewerCfg = ViewerCfg()
    # viewer.eye = (0.6, 0.1, 0.025)
    # viewer.lookat = (-1.2, -2.0, -0.2)
    viewer.eye = (0.55, -0.06, 0.025)
    viewer.lookat = (-4.8, 6.0, -0.2)

    debug_vis = True

    ui_window_class_type = CustomEnvWindow

    decimation = 1
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physx=PhysxCfg(
            enable_ccd=True,  # needed for more stable ball_rolling
            # bounce_threshold_velocity=10000,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=5.0,
            dynamic_friction=5.0,
            restitution=0.0,
        ),
        render=RenderCfg(enable_translucency=True),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=1.5,
        replicate_physics=True,
        lazy_sensor_update=True,  # only update sensors when they are accessed
    )

    # Ground-plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, 0)),
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
        ),
    )

    # light
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # Robot and sensor configs will be set dynamically based on sensor_type
    robot: ArticulationCfg = None
    shapes: dict = create_shapes_cfg()
    gsmini: GelSightMiniCfg = None
    sensor_type: str = "gsmini"  # "gsmini" or "digit"

    ik_controller_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")

    main_pose = [0.5, 0.0, 0.02, 1, 0, 0, 0]

    # some filler values, needed for DirectRLEnv class
    episode_length_s = 0
    action_space = 0
    observation_space = 0
    state_space = 0


class BallRollingEnv(DirectRLEnv):
    cfg: BallRollingEnvCfg

    def __init__(self, cfg: BallRollingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # --- for IK ---
        # create the differential IK controller
        self._ik_controller = DifferentialIKController(
            cfg=self.cfg.ik_controller_cfg, num_envs=self.num_envs, device=self.device
        )
        # Obtain the frame index of the end-effector
        body_ids, body_names = self._robot.find_bodies("panda_hand")
        # save only the first body index
        self._body_idx = body_ids[0]
        self._body_name = body_names[0]

        # For a fixed base robot, the frame index is one less than the body index.
        # This is because the root body is not included in the returned Jacobians.
        self._jacobi_body_idx = self._body_idx - 1
        # self._jacobi_joint_ids = self._joint_ids # we take every joint

        # ee offset w.r.t panda hand -> based on the asset
        # Different sensors have different offsets due to different geometry
        if self.cfg.sensor_type == "digit":
            # DIGIT sensor has a different geometry, adjust offset for proper alignment
            self._offset_pos = torch.tensor([0.0, 0.0, 0.132], device=self.device).repeat(self.num_envs, 1)
        else:  # gsmini
            self._offset_pos = torch.tensor([0.0, 0.0, 0.131], device=self.device).repeat(self.num_envs, 1)
        self._offset_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        # ---

        # create buffer to store actions (= ik_commands)
        self.ik_commands = torch.zeros((self.num_envs, self._ik_controller.action_dim), device=self.device)
        # self.ik_commands[:, 3:] = torch.tensor([0,1,0,0],device=self.device)

        self.step_count = 0

        self.goal_prim_view = None

        self.main_pose = torch.tensor([self.cfg.main_pose], device=self.device)

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        for obj_name, cfg in self.cfg.shapes.items():
            self.scene.rigid_objects[obj_name] = RigidObject(cfg)

        self.object = list(self.scene.rigid_objects.values())[0]

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        
        # Set offset based on sensor type
        if self.cfg.sensor_type == "digit":
            ee_offset_pos = (0.0, 0.0, 0.161)
        else:  # gsmini
            ee_offset_pos = (0.0, 0.0, 0.131)
        
        ee_frame_cfg = FrameTransformerCfg(
            prim_path="/World/envs/env_.*/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="/World/envs/env_.*/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=ee_offset_pos,
                    ),
                ),
            ],
        )

        # sensors
        self._ee_frame = FrameTransformer(ee_frame_cfg)
        self.scene.sensors["ee_frame"] = self._ee_frame

        self.gsmini = GelSightSensor(self.cfg.gsmini)
        self.scene.sensors["gsmini"] = self.gsmini

        # Modify camera parameters to reduce wide-angle effect
        self._adjust_camera_parameters()

        # Spawn AssetBase objects manually
        ground = self.cfg.ground
        ground.spawn.func(
            ground.prim_path,
            ground.spawn,
            translation=ground.init_state.pos,
            orientation=ground.init_state.rot,
        )

        # goal marker
        VisualCuboid(
            prim_path="/Goal",
            size=0.01,
            position=np.array([0.5, 0.0, 0.021]),
            orientation=np.array([0, 1, 0, 0]),
            visible=False,
        )

        # just for visualizen what the main prim is
        VisualCuboid(
            prim_path="/Visuals/main_area",
            size=0.02,
            position=np.array([0.5, 0.0, -0.005]),
            color=np.array([255.0, 0.0, 0.0]),
        )

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # MARK: pre-physics step calls

    def _pre_physics_step(self, actions: torch.Tensor):
        self._ik_controller.set_command(self.ik_commands)

    def _apply_action(self):
        # obtain quantities from simulation
        ee_pos_curr_b, ee_quat_curr_b = self._compute_frame_pose()
        joint_pos = self._robot.data.joint_pos[:, :]

        # compute the delta in joint-space
        if ee_pos_curr_b.norm() != 0:
            jacobian = self._compute_frame_jacobian()
            joint_pos_des = self._ik_controller.compute(ee_pos_curr_b, ee_quat_curr_b, jacobian, joint_pos)
        else:
            joint_pos_des = joint_pos.clone()
        self._robot.set_joint_position_target(joint_pos_des)

        self.step_count += 1

    # post-physics step calls

    # MARK: dones
    def _get_dones(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:  # which environment is done
        pass

    # MARK: rewards
    def _get_rewards(self) -> torch.Tensor:
        pass

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)

        # set goal pos
        if self.goal_prim_view is not None:
            # move goal back to main area
            goal_pos = self.main_pose[:, :3]
            goal_pos[:, 2] += 0.001

            goal_orient = torch.tensor([[0, 1, 0, 0]], device=self.device)
            self.goal_prim_view.set_world_poses(positions=goal_pos, orientations=goal_orient)

        # reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]

        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    # MARK: observations
    def _get_observations(self) -> dict:
        pass

    """
    Helper Functions for IK control (from task_space_actions.py of IsaacLab).
    """

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
        """Computes the pose of the target frame in the root frame.

        Returns:
            A tuple of the body's position and orientation in the root frame.
        """
        # obtain quantities from simulation
        ee_pos_w = self._robot.data.body_link_pos_w[:, self._body_idx]
        ee_quat_w = self._robot.data.body_link_quat_w[:, self._body_idx]
        root_pos_w = self._robot.data.root_link_pos_w
        root_quat_w = self._robot.data.root_link_quat_w
        # compute the pose of the body in the root frame
        ee_pose_b, ee_quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
        # account for the offset
        # if self.cfg.body_offset is not None:
        ee_pose_b, ee_quat_b = math_utils.combine_frame_transforms(
            ee_pose_b, ee_quat_b, self._offset_pos, self._offset_rot
        )

        return ee_pose_b, ee_quat_b

    def _compute_frame_jacobian(self):
        """Computes the geometric Jacobian of the target frame in the root frame.

        This function accounts for the target frame offset and applies the necessary transformations to obtain
        the right Jacobian from the parent body Jacobian.
        """
        # read the parent jacobian
        jacobian = self.jacobian_b

        # account for the offset
        # if self.cfg.body_offset is not None:
        # Modify the jacobian to account for the offset
        # -- translational part
        # v_link = v_ee + w_ee x r_link_ee = v_J_ee * q + w_J_ee * q x r_link_ee
        #        = (v_J_ee + w_J_ee x r_link_ee ) * q
        #        = (v_J_ee - r_link_ee_[x] @ w_J_ee) * q
        jacobian[:, 0:3, :] += torch.bmm(-math_utils.skew_symmetric_matrix(self._offset_pos), jacobian[:, 3:, :])
        # -- rotational part
        # w_link = R_link_ee @ w_ee
        jacobian[:, 3:, :] = torch.bmm(math_utils.matrix_from_quat(self._offset_rot), jacobian[:, 3:, :])

        return jacobian

    def _adjust_camera_parameters(self):
        """Adjust camera parameters to reduce wide-angle effect.
        
        Original USD values:
        - GelSight Mini: focalLength=20.0, horizontalAperture=14.3, verticalAperture=18.6
        - DIGIT: focalLength=26.0, horizontalAperture=20.0, verticalAperture=18.0
        
        To adjust FOV:
        - Smaller aperture = narrower FOV (less wide-angle)
        - Larger aperture = wider FOV (more wide-angle)
        - Larger focal length = narrower FOV
        """
        from pxr import Usd, UsdGeom
        
        # Get the camera prim path
        if self.cfg.sensor_type == "digit":
            camera_path = "/World/envs/env_0/Robot/digit_case/Camera"
            # Original DIGIT values
            original_focal = 26.0
            original_h_aperture = 20.0
            original_v_aperture = 18.0
        else:  # gsmini
            camera_path = "/World/envs/env_0/Robot/gelsight_mini_case/Camera"
            # Original GelSight Mini values
            original_focal = 20.0
            original_h_aperture = 14.3
            original_v_aperture = 18.6
        
        # Get the stage and camera prim
        stage = omni.usd.get_context().get_stage()
        camera_prim = stage.GetPrimAtPath(camera_path)
        
        if camera_prim.IsValid():
            camera = UsdGeom.Camera(camera_prim)
            
            # Set to original values (no adjustment by default)
            # Uncomment and modify these lines to adjust the camera view:
            
            # camera.GetFocalLengthAttr().Set(original_focal)
            # camera.GetHorizontalApertureAttr().Set(original_h_aperture)
            # camera.GetVerticalApertureAttr().Set(original_v_aperture)
            
            # Example: Reduce wide-angle effect by 30%
            #camera.GetHorizontalApertureAttr().Set(original_h_aperture * 0.82)
            #camera.GetVerticalApertureAttr().Set(original_v_aperture * 0.82)
            
            print(f"[Info] Camera parameters at {camera_path}")
            print(f"       Original values: focal={original_focal}, h_aperture={original_h_aperture}, v_aperture={original_v_aperture}")
            print(f"       (No adjustment applied - using original values)")
        else:
            print(f"[Warning] Camera prim not found at {camera_path}")


def run_simulator(env: BallRollingEnv):
    """Runs the simulation loop."""
    global current_tactile_image
    
    # for convenience, we directly turn on debug_vis
    if env.cfg.gsmini.debug_vis:
        for data_type in env.cfg.gsmini.data_types:
            env.gsmini._prim_view.prims[0].GetAttribute(f"debug_{data_type}").Set(True)

    # for saving tactile images
    # timestamp = f"{datetime.datetime.now():%Y-%m-%d-%H_%M_%S}"
    # output_dir = Path(__file__).parent.resolve()
    # file_name = str(output_dir) + f"/envs_{env.num_envs}_{timestamp}.txt"

    print(f"Starting simulation with {env.num_envs} envs")
    
    # Create OpenCV window for tactile image display
    window_name = "Tactile RGB (Click to Save)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 480)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    print(f"[INFO] OpenCV window created: '{window_name}'")
    print(f"[INFO] Click on the window to save images to: {SAVE_DIR}")
    print(f"[INFO] Press 'q' in OpenCV window to quit")

    env.reset()
    env.goal_prim_view = XFormPrim(prim_paths_expr="/Goal", name="Goal", usd=True)

    # Simulation loop
    while simulation_app.is_running():
        # perform physics step
        env._pre_physics_step(None)
        env._apply_action()
        env.scene.write_data_to_sim()
        env.sim.step(render=False)

        positions, orientations = env.goal_prim_view.get_world_poses()
        env.ik_commands[:, :3] = positions - env.scene.env_origins
        env.ik_commands[:, 3:] = orientations

        # update isaac buffers() -> also updates sensors
        env.scene.update(dt=env.physics_dt)
        # render scene for cameras (used by sensor)
        env.sim.render()
        
        # Get and display tactile RGB image
        try:
            # Get tactile_rgb data from sensor
            if "tactile_rgb" in env.gsmini.data.output:
                tactile_rgb = env.gsmini.data.output["tactile_rgb"][0].cpu().numpy()
                
                # Convert from RGB to BGR for OpenCV display
                if tactile_rgb.shape[-1] == 3:
                    tactile_bgr = cv2.cvtColor(tactile_rgb, cv2.COLOR_RGB2BGR)
                else:
                    tactile_bgr = tactile_rgb
                
                # Convert to uint8 if needed
                if tactile_bgr.dtype != np.uint8:
                    if tactile_bgr.max() <= 1.0:
                        tactile_bgr = (tactile_bgr * 255).astype(np.uint8)
                    else:
                        tactile_bgr = tactile_bgr.astype(np.uint8)
                
                # Update global variable for saving
                current_tactile_image = tactile_bgr.copy()
                
                # Display the image
                cv2.imshow(window_name, tactile_bgr)
                
        except Exception as e:
            print(f"[WARNING] Error displaying tactile image: {e}")
        
        # Check for key press (1ms wait)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[INFO] 'q' pressed, exiting...")
            break

    # )
    #         env.gsmini.update(dt=env.physics_dt, force_recompute=True)

    # Cleanup
    cv2.destroyAllWindows()
    env.close()

    pynvml.nvmlShutdown()


def main():
    """Main function."""
    # Get sensor type from CLI arguments
    sensor_type = args_cli.sensor_type.lower()
    print(f"[INFO]: Using sensor type: {sensor_type.upper()}")
    
    # Define simulation env
    env_cfg = BallRollingEnvCfg()
    
    # Set robot and sensor configurations based on sensor type
    env_cfg.sensor_type = sensor_type
    env_cfg.robot = create_robot_cfg(sensor_type)
    env_cfg.gsmini = create_sensor_cfg(sensor_type, env_cfg.shapes)
    
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.gsmini.debug_vis = args_cli.debug_vis

    experiment = BallRollingEnv(env_cfg)

    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(env=experiment)


if __name__ == "__main__":
    try:
        # run the main execution
        main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        # close sim apply
        simulation_app.close()
