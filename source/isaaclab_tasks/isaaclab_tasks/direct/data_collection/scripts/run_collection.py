from __future__ import annotations

import argparse

import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Rope data collection")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_episodes", type=int, default=200)
parser.add_argument("--output_dir", type=str, default="/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/direct/data_collection/data/rope_collection")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import sys
sys.path.append("/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/direct/data_collection")
from data_collection_env_cfg import RopeCollectionEnv, RopeCollectionEnvCfg
from data_collection_utils import DataWriter, LabelGenerator

import omni.usd
from pxr import Usd

def main() -> None:
    cfg = RopeCollectionEnvCfg()
    cfg.scene.num_envs = args.num_envs

    writer = DataWriter(output_dir=args.output_dir)
    env = RopeCollectionEnv(cfg)

    stage = omni.usd.get_context().get_stage()

    label_gen = LabelGenerator(env.scene["rope"], env.scene["camera"])
    label_gen.build_map()

    settle_steps = env._settle_steps
    default_joint_pos = env.scene["robot"].data.default_joint_pos
    print(f"default: {default_joint_pos}")
    dummy_action = default_joint_pos.clone()

    for episode in range(args.num_episodes):
        env.reset()

        for _ in range(settle_steps):
            env.step(dummy_action)

        for _ in range(cfg.collect_per_episode):
            env.step(dummy_action)

            depth      = env.scene["camera"].data.output["depth"][..., 0]            # (N, H, W)
            seg        = env.scene["camera"].data.output["instance_segmentation_fast"]
            link_pos   = env.scene["rope"].data.body_pos_w                            # (N, L, 3)
            link_label = label_gen.seg_to_link_label(seg)                             # (N, H, W)
            knot_flag  = env.get_knot_flags()                                         # (N,)

            writer.write(depth, link_label, link_pos, knot_flag, meta={"episode": episode})

    writer.close()
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()