"""Pair dataset for the HD->MC Stylizer: (reference, prompt, MC target).

Returns (x_target, prompt_str, aux) where aux carries the reference tensor and
the tileable flag, matching the batch convention used by Trainer (so the text
encode pipeline and the existing collate work unchanged).
"""
import json
import os
from pathlib import Path

import numpy as np
import torch


class MmapPairDataset(torch.utils.data.Dataset):
    def __init__(self, ref_mmap, target_mmap, metadata, splits, split="train",
                 ref_size=64, target_size=32, rgba_mode="premultiplied",
                 prompt_col="prompt", ref_premultiplied=False, zero_reference=False,
                 shuffle_reference=False, seed=0):
        self.ref_size = int(ref_size)
        self.target_size = int(target_size)
        self.rgba_mode = rgba_mode
        self.ref_premultiplied = bool(ref_premultiplied)
        self.zero_reference = bool(zero_reference)
        self.shuffle_reference = bool(shuffle_reference)
        self.rng = np.random.RandomState(seed)

        split_map = json.loads(Path(splits).read_text())
        self.index = list(split_map.get(split, []))
        import pandas as pd

        self.df = pd.read_parquet(metadata)
        self.n = len(self.df)
        self.prompt_col = prompt_col
        self.ref = self._mmap(ref_mmap, (self.n, self.ref_size, self.ref_size, 4))
        self.target = self._mmap(target_mmap, (self.n, self.target_size, self.target_size, 4))

    @staticmethod
    def _mmap(path, shape):
        expected = int(np.prod(shape))
        actual = os.path.getsize(path)
        if actual != expected:
            raise ValueError(f"{path}: {actual} bytes, expected {expected} for {shape}")
        return np.memmap(path, dtype=np.uint8, mode="r", shape=shape)

    def __len__(self):
        return len(self.index)

    def _to_tensor(self, arr):
        t = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).float() / 127.5 - 1.0
        if self.rgba_mode == "premultiplied":
            from data.rgba import premultiply_rgba_torch

            t = premultiply_rgba_torch(t[None])[0]
        return t

    def __getitem__(self, i):
        idx = self.index[i]
        x = self._to_tensor(np.asarray(self.target[idx]))
        ref_idx = idx
        if self.shuffle_reference:
            ref_idx = int(self.index[self.rng.randint(len(self.index))])
        ref = self._to_tensor(np.asarray(self.ref[ref_idx]))
        if self.zero_reference:
            ref = torch.zeros_like(ref)
        elif self.ref_premultiplied:
            from data.rgba import premultiply_rgba_torch

            ref = premultiply_rgba_torch(ref[None])[0]
        prompt = str(self.df[self.prompt_col].iloc[idx])
        aux = {
            "reference": ref,
            "tileable": torch.tensor(bool(self.df["tileable"].iloc[idx]), dtype=torch.bool),
        }
        return x, prompt, aux
