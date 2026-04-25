
import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg, RemotizedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

TONG_SYSTEM_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/workspace/isaaclab/source/isaaclab_assets/isaaclab_assets/robots/usd/tong_system/tong_system.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=True,
            max_depenetration_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "r_shoulder_pan_joint": 0.8200,
            "r_shoulder_lift_joint": -1.6216,
            "r_elbow_joint": -1.1622,
            "r_wrist_1_joint": -2.0251,
            "r_wrist_2_joint": 1.1879,
            "r_wrist_3_joint": -0.3475,
            "r_move_joint": 0.0,
            "l_shoulder_pan_joint": -0.7491,
            "l_shoulder_lift_joint": -1.3549,
            "l_elbow_joint": 1.0856,
            "l_wrist_1_joint": -0.9650,
            "l_wrist_2_joint": -1.2006,
            "l_wrist_3_joint": 0.2964,
            "l_move_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "actuator": DelayedPDActuatorCfg(
            joint_names_expr=[
                "r_shoulder_pan_joint",
                "r_shoulder_lift_joint",
                "r_elbow_joint",
                "r_wrist_1_joint",
                "r_wrist_2_joint",
                "r_wrist_3_joint",
                "r_move_joint",
                "l_shoulder_pan_joint",
                "l_shoulder_lift_joint",
                "l_elbow_joint",
                "l_wrist_1_joint",
                "l_wrist_2_joint",
                "l_wrist_3_joint",
                "l_move_joint",
            ],
            stiffness=100.0,
            damping=1.0,
        ),
    },
)