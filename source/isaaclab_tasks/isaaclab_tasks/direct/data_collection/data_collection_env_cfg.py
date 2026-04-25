from __future__ import annotations

import math
import torch
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, Articulation, RigidObject
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg, TiledCamera
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.wrappers import MultiAssetSpawnerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from isaaclab_assets.objects.link_rod import LinkedRodObjectCfg

from isaaclab_assets.robots.tong_system import TONG_SYSTEM_CFG

TABLE_HEIGHT: float = 0.8

@configclass
class RopeCollectionSceneCfg(InteractiveSceneCfg):
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 0.0]),
        spawn=GroundPlaneCfg(),
    )

    robot = TONG_SYSTEM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.6, 0, 0.0]),
        spawn=UsdFileCfg(usd_path=f"/workspace/isaaclab/source/isaaclab_assets/isaaclab_assets/robots/usd/table/table.usd"),
    )

    rope = LinkedRodObjectCfg(
        prim_path="{ENV_REGEX_NS}/Rope",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=[0.6, 0.0, 1.0],
            rot=[0, 1, 0, 0],
            joint_pos={"rod_joint_1": 0.0, "rod_joint_2": 0.0},
        ),
        link_length=0.200,
        link_radius=0.008,
        joint_type="revolute",
    )

    camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/neck_pitch_link/right_cam_link/right_view_link/camera",
        update_period=0.033,
        height=720,
        width=1280,
        data_types=["depth", "instance_segmentation_fast"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 3.0),
            # clipping_range=(0.01, 30.0), # TODO: fix hardcoded max depth value
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(0.8433914458128857, 0.5372996083468239, 0.0, 0.0), # 65.0 deg
            # rot=(1.0, 0.0, 0.0, 0.0), # 0.0 deg
            convention="ros",
        ),
    )

@configclass
class RopeCollectionEnvCfg(DirectRLEnvCfg):
    episode_length_s: float = 5.0
    decimation: int = 2
    action_space: int = 14
    observation_space: int = 1
    state_space: int = 0

    settle_time_s: float = 2.0
    collect_per_episode: int = 5
    table_height: float = TABLE_HEIGHT

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    scene: RopeCollectionSceneCfg = RopeCollectionSceneCfg(num_envs=64, env_spacing=4.0)

class RopeCollectionEnv(DirectRLEnv):
    cfg: RopeCollectionEnvCfg

    def __init__(self, cfg: RopeCollectionEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._settle_steps = int(cfg.settle_time_s / (cfg.sim.dt * cfg.decimation))
        self._default_link_length: float | None = None
        

    def _setup_scene(self) -> None:
        self._robot  = self.scene["robot"]
        self._rope   = self.scene["rope"]
        self._table  = self.scene["table"]
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        self._camera = self.scene["camera"]

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._robot.set_joint_position_target(
            self._robot.data.default_joint_pos
        )

    def _apply_action(self) -> None:
        self._robot.write_data_to_sim()

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        terminated = torch.zeros_like(truncated)
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        default_pos = self._robot.data.default_joint_pos[env_ids]
        default_vel = self._robot.data.default_joint_vel[env_ids]
        self._robot.write_joint_state_to_sim(default_pos, default_vel, env_ids=env_ids)
        self._robot.write_data_to_sim()
        super()._reset_idx(env_ids)

    def _get_observations(self) -> dict:
        return {"policy": torch.zeros((self.num_envs, 1), device=self.device)}

    def get_knot_flags(self) -> torch.Tensor:
        """Non-adjacent link proximity heuristic for knot detection."""
        link_pos = self._rope.data.body_pos_w  # (N, L, 3)
        N, L, _ = link_pos.shape

        if L < 4:
            return torch.zeros(N, dtype=torch.bool, device=self.device)

        adj_dists = torch.norm(link_pos[:, 1:] - link_pos[:, :-1], dim=-1)  # (N, L-1)
        if self._default_link_length is None:
            self._default_link_length = adj_dists.mean().item()

        threshold = self._default_link_length * 1.5

        # Links separated by ≥3 hops shouldn't be geometrically close unless knotted
        p0 = link_pos[:, :-3]   # (N, L-3, 3)
        p1 = link_pos[:, 3:]    # (N, L-3, 3)
        close = torch.norm(p0 - p1, dim=-1) < threshold  # (N, L-3)
        return close.any(dim=-1)