"""
ランダム配置の責務のみ持つ
EventManagerから呼ばれる
"""
from __future__ import annotations

import math

import torch
from isaaclab.envs import ManagerBasedEnv


def random_yaw_orientation(N: int, device=None) -> torch.Tensor:
    """ランダムなyaw回転のクォータニオン (N, 4) wxyz"""
    yaw = torch.empty(N, device=device).uniform_(0.0, 2.0 * math.pi)
    q = torch.zeros(N, 4, device=device)
    q[:, 0] = torch.cos(yaw * 0.5)  # w
    q[:, 3] = torch.sin(yaw * 0.5)  # z
    return q


def randomize_rope_on_table(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    table_height: float,
    pos_range: tuple[float, float],
) -> None:
    rope = env.scene["rope"]
    N = len(env_ids)
    device = env_ids.device

    pos = rope.data.default_root_state[env_ids, :3].clone()
    pos[:, 0] += torch.empty(N, device=device).uniform_(*pos_range)
    pos[:, 1] += torch.empty(N, device=device).uniform_(*pos_range)
    pos[:, 2]  = table_height + 0.1

    orientation = random_yaw_orientation(N, device=device)

    rope.write_root_pose_to_sim(
        torch.cat([pos, orientation], dim=-1), env_ids
    )