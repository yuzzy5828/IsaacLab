from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch import Tensor

class DataWriter:
    def __init__(self, output_dir: str, chunk_size: int = 2):
        self._path = Path(output_dir)
        self._path.mkdir(parents=True, exist_ok=True)
        self._chunk_size = chunk_size
        self._buffer: list[dict[str, np.ndarray]] = []
        self._file_idx = 0

    def write(
        self,
        depth: Tensor,       # (N, H, W)
        link_label: Tensor,  # (N, H, W)
        link_pos: Tensor,    # (N, L, 3)
        knot_flag: Tensor,   # (N,)
        meta: dict,
    ) -> None:
        self._buffer.append({
            "depth":      depth.cpu().float().numpy(),
            "link_label": link_label.cpu().to(torch.int16).numpy(),
            "link_pos":   link_pos.cpu().float().numpy(),
            "knot_flag":  knot_flag.cpu().bool().numpy(),
        })
        if len(self._buffer) >= self._chunk_size:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        out = self._path / f"chunk_{self._file_idx:06d}.h5"
        with h5py.File(out, "w") as f:
            for key in ("depth", "link_label", "link_pos", "knot_flag"):
                arr = np.concatenate([b[key] for b in self._buffer], axis=0)
                f.create_dataset(key, data=arr, compression="gzip", compression_opts=4)
        self._file_idx += 1
        self._buffer.clear()

    def close(self) -> None:
        self._flush()