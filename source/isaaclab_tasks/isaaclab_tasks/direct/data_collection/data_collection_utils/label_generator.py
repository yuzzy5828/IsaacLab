from __future__ import annotations

import torch
from torch import Tensor


def build_prim_to_link_map(articulation, env_ids) -> dict[str, int]:
    """body名 → linkインデックスのマッピング構築 (env非依存)"""
    return {name: idx for idx, name in enumerate(articulation.data.body_names)}


def seg_to_link_label(
    seg_image: Tensor,
    prim_to_link: dict[str, int],
    id_to_labels: list[dict] | dict,
) -> Tensor:
    if seg_image.dim() == 4:
        seg_image = seg_image[..., 0]

    N, H, W = seg_image.shape
    result = torch.full((N, H, W), -1, dtype=torch.int16, device=seg_image.device)

    for env_idx in range(N):
        # --- 修正箇所 ---
        # info[env_idx] は直接 ID マップではなく、メタデータを含む辞書です。
        # 実際の ID -> ラベルのマップは "id_to_labels" キーの中にあります。
        full_info = id_to_labels[env_idx] if isinstance(id_to_labels, list) else id_to_labels
        
        # 'id_to_labels' キーが存在するか確認して取得
        mapping = full_info.get("id_to_labels", full_info) 

        for inst_id, info in mapping.items():
            # info が辞書でない場合（背景など）や 'class' キーがない場合を考慮
            if not isinstance(info, dict) or "class" not in info:
                continue
            
            prim_path = info["class"]
            body_name = prim_path.rsplit("/", 1)[-1]
            link_idx = prim_to_link.get(body_name, -1)
            
            if link_idx < 0:
                continue
            
            # inst_id を整数に変換してマスク適用
            result[env_idx][seg_image[env_idx] == int(inst_id)] = link_idx

    return result


class LabelGenerator:
    def __init__(self, rope, camera):
        self._rope = rope
        self._camera = camera
        self._prim_to_link: dict[str, int] = {}

    def build_map(self, env_ids: Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.tensor([0])
        self._prim_to_link = build_prim_to_link_map(self._rope, env_ids)

    def seg_to_link_label(self, seg_image: Tensor) -> Tensor:
        if not self._prim_to_link:
            self.build_map()
        id_to_labels = self._camera.data.info["instance_segmentation_fast"]
        return seg_to_link_label(seg_image, self._prim_to_link, id_to_labels)