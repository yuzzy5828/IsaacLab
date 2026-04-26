from __future__ import annotations

import torch
from torch import Tensor
import numpy as np
import re
import ast

def build_prim_to_link_map(articulation, env_ids) -> dict[str, int]:
    mapping = {}
    for name in articulation.data.body_names:
        # "rod_link_0" -> 0, "rod_link_1" -> 1 のように数字を抽出
        match = re.search(r'rod_link_(\d+)', name)
        if match:
            mapping[name] = int(match.group(1))
    return mapping

def seg_to_link_label(
    seg_image: Tensor,
    prim_to_link: dict[str, int],
    id_to_labels: list[dict] | dict,
) -> Tensor:
    # 画像が (N, H, W, 4) のRGBAか (N, H, W, 1) のIDマップかを判定
    is_rgba = False
    if seg_image.dim() == 4:
        if seg_image.shape[-1] == 4:
            is_rgba = True
        else:
            seg_image = seg_image[..., 0]  # (N, H, W, 1) -> (N, H, W)

    N, H, W = seg_image.shape[:3]
    result = torch.full((N, H, W), -1, dtype=torch.int16, device=seg_image.device)

    for env_idx in range(N):
        full_info = id_to_labels[env_idx] if isinstance(id_to_labels, list) else id_to_labels

        # ① 修正: "idToLabels" キーを最優先で取得
        mapping = full_info.get("idToLabels") or full_info.get("idToSemantics") or full_info.get("id_to_labels", {})

        for inst_id_str, info in mapping.items():
            if not isinstance(info, dict):
                continue

            class_name = info.get("class", "")
            if not class_name:
                continue

            # class名がbody_namesに直接マッチするか確認
            link_idx = prim_to_link.get(class_name, -1)

            # マッチしない場合はパス末尾で試みる
            if link_idx < 0:
                body_name = class_name.rsplit("/", 1)[-1]
                link_idx = prim_to_link.get(body_name, -1)

            if link_idx < 0:
                continue

            r = g = b = a = None
            int_id = None

            # ② 修正: 文字列 "(R, G, B, A)" を Python のタプルに変換
            try:
                if isinstance(inst_id_str, str) and inst_id_str.startswith("("):
                    r, g, b, a = ast.literal_eval(inst_id_str)
                elif isinstance(inst_id_str, tuple):
                    r, g, b, a = inst_id_str
                else:
                    int_id = int(inst_id_str)
            except Exception:
                continue

            if is_rgba and r is not None:
                # RGBA画像として出力されている場合
                mask = (
                    (seg_image[env_idx, ..., 0] == r) &
                    (seg_image[env_idx, ..., 1] == g) &
                    (seg_image[env_idx, ..., 2] == b)
                )
                result[env_idx][mask] = link_idx
            else:
                if r is not None:
                    # ID画像 (N, H, W) の場合
                    # Replicatorは RGBAを 32ビット整数にパックするため、Python(int32)では負の数になることがある
                    val = np.uint32(r + (g << 8) + (b << 16) + (a << 24))
                    int_id_with_alpha = int(val.astype(np.int32))
                    int_id_no_alpha = r + (g << 8) + (b << 16)
                    
                    # 念のため、Alphaあり・なし両方でマッチングする
                    mask = (seg_image[env_idx] == int_id_with_alpha) | (seg_image[env_idx] == int_id_no_alpha)
                    result[env_idx][mask] = link_idx
                elif int_id is not None:
                    result[env_idx][seg_image[env_idx] == int_id] = link_idx

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
        id_to_labels = self._camera.data.info["semantic_segmentation"]
        return seg_to_link_label(seg_image, self._prim_to_link, id_to_labels)