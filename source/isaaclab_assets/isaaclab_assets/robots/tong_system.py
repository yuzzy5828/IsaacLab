
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
            ".*_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "actuator": DelayedPDActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=100.0,
            damping=1.0,
        ),
    },
)