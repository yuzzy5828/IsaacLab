# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import math

from isaaclab.assets import RigidObjectCfg, ArticulationCfg, DeformableObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_assets.objects.link_rod import LinkedRodObjectCfg
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.config.tong_system.lift_tong_system_env_cfg import LiftCustomTableEnvCfg, LiftCustomTableWithDepthEnvCfg
from isaaclab_tasks.manager_based.manipulation.lift.config.tong_system.lift_tong_system_env_cfg import LinkedRodEventCfg, DeformableObjectEventCfg, LinkedRodTerminationsCfg, DeformableObjectTerminationsCfg

from isaaclab_assets.robots.tong_system import TONG_SYSTEM_CFG

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip


@configclass
class TongSystemCubeLiftEnvCfg(LiftCustomTableEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set TongSystem as robot
        self.scene.robot = TONG_SYSTEM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (TongSystem)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                ".*_shoulder_.*_joint",
                ".*_elbow_joint",
                ".*_wrist_.*_joint",
            ],
            scale=0.5,
            use_default_offset=True,
        )

        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*_move_joint",],
            open_command_expr={".*_move_joint": 0.044},
            close_command_expr={".*_move_joint": 0.0},
        )

        # Set the body name for the end effector
        self.commands.object_pose.body_name = ".*_move_link"
        self.commands.object_pose.ranges.pitch = (math.pi / 2, math.pi / 2)

        # Set Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, 0, 0.8], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/r_fixed_finger_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/r_fixed_finger_link",
                    name="end_effector",
                ),
            ],
        )

@configclass
class TongSystemCubeLiftWithDepthEnvCfg(LiftCustomTableWithDepthEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set TongSystem as robot
        self.scene.robot = TONG_SYSTEM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (TongSystem)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                ".*_shoulder_.*_joint",
                ".*_elbow_joint",
                ".*_wrist_.*_joint",
            ],
            scale=0.5,
            use_default_offset=True,
        )

        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*_move_joint",],
            open_command_expr={".*_move_joint": 0.044},
            close_command_expr={".*_move_joint": 0.0},
        )

        # camera settings
        self.scene.camera.prim_path = "{ENV_REGEX_NS}/Robot/neck_pitch_link/right_cam_link/camera"

        # Set the body name for the end effector
        self.commands.object_pose.body_name = ".*_move_link"
        self.commands.object_pose.ranges.pitch = (math.pi / 2, math.pi / 2)

        # Set Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, 0, 0.8], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/r_fixed_finger_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/r_fixed_finger_link",
                    name="end_effector",
                ),
            ],
        )


##
# 3リンク棒オブジェクト用 Lift Cfg（深度なし）
##

@configclass
class TongSystemLinkedRodLiftEnvCfg(TongSystemCubeLiftEnvCfg):

    events: LinkedRodEventCfg = LinkedRodEventCfg()
    terminations: LinkedRodTerminationsCfg = LinkedRodTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.object = LinkedRodObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            joint_positions=[
                (-0.04, 0.00, 0.00),
                (0.01, 0.06, 0.00),
                (0.03, 0.09, 0.00),
                (0.04, 0.12, 0.00),
            ],
            link_radius=0.004,
            joint_type="revolute",
        )

        self.scene.replicate_physics = False


##
# 3リンク棒オブジェクト用 Lift Cfg（深度あり）
##

@configclass
class TongSystemLinkedRodLiftWithDepthEnvCfg(TongSystemCubeLiftWithDepthEnvCfg):
    """深度カメラ付き・固定3リンク棒リフト環境。"""

    events: LinkedRodEventCfg = LinkedRodEventCfg()
    terminations: LinkedRodTerminationsCfg = LinkedRodTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.object = LinkedRodObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            joint_positions=[
                (-0.04, 0.00, 0.00),
                (0.01, 0.06, 0.00),
                (0.03, 0.09, 0.00),
                (0.04, 0.12, 0.00),
            ],
            link_radius=0.004,
            joint_type="revolute",
        )

        self.scene.replicate_physics = False

@configclass
class TongSystemDeformableObjectLiftWithDepthEnvCfg(TongSystemCubeLiftWithDepthEnvCfg):
    events: DeformableObjectEventCfg = DeformableObjectEventCfg()
    terminations: DeformableObjectTerminationsCfg = DeformableObjectTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Set Deformable Object
        self.scene.object = DeformableObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=DeformableObjectCfg.InitialStateCfg(pos=[0.5, 0.0, 0.8], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/DeformableTube/tube.usd",
                scale=(1.5, 1.5, 1.5),
            )
        )
    
        self.scene.replicate_physics = False

@configclass
class TongSystemCubeLiftEnvCfg_PLAY(TongSystemCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
