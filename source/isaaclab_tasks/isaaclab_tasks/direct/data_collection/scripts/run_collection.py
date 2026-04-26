from __future__ import annotations

import argparse

import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Rope data collection")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_episodes", type=int, default=10)
parser.add_argument("--output_dir", type=str, default="/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/direct/data_collection/data/rope_collection")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import re
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

    # body_namesの順番は物理エンジンの内部で決まるため，末端からラベルづけするように変更
    body_names = env.scene["rope"].data.body_names
    order = []
    for i, name in enumerate(body_names):
        match = re.search(r'rod_link_(\d+)', name)
        if match:
            order.append((int(match.group(1)), i))
    
    # リンク番号でソートし、元のインデックス配列を取得
    order.sort(key=lambda x: x[0])
    sort_indices = [x[1] for x in order]

    settle_steps = env._settle_steps
    default_joint_pos = env.scene["robot"].data.default_joint_pos
    print(f"default: {default_joint_pos}")
    dummy_action = default_joint_pos.clone()

    for episode in range(args.num_episodes):
        env.reset()

        for _ in range(settle_steps):
            env.step(dummy_action)

        for _ in range(cfg.collect_per_episode):
            env_origins = env.scene.env_origins
            env.step(dummy_action)

            depth      = env.scene["camera"].data.output["depth"][..., 0]            # (N, H, W)
            seg = env.scene["camera"].data.output["semantic_segmentation"]
            id_to_labels = env.scene["camera"].data.info["semantic_segmentation"]

            # env_0の内容を確認
            print("=== body_names ===")
            print(env.scene["rope"].data.body_names)

            print("=== id_to_labels[0] ===")
            info0 = id_to_labels[0] if isinstance(id_to_labels, list) else id_to_labels
            mapping = info0.get("id_to_labels", info0)
            for inst_id, info in list(mapping.items())[:10]:
                print(f"  id={inst_id}: {info}")
            link_pos_w = env.scene["rope"].data.body_pos_w
            link_pos_w = link_pos_w[:, sort_indices, :]                               # 末端からの順番にソート
            link_pos_l = link_pos_w - env_origins.unsqueeze(1)                        # (N, L, 3)
            link_label = label_gen.seg_to_link_label(seg)                             # (N, H, W)
            knot_flag  = env.get_knot_flags()                                         # (N,)

            writer.write(depth, link_label, link_pos_l, knot_flag, meta={"episode": episode})

    writer.close()
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()