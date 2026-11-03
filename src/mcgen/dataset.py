"""Catalog-backed loading. No augmentation or replay can be enabled implicitly."""
import numpy as np
import torch
from collections import defaultdict

from .images import content_hash, encode, native_grid, reference


class CatalogDataset:
    def __init__(self, records, size, ref_size, split, asset_type="all", task="joint", alpha_mode="continuous", style="", target_scope="all"):
        self.size, self.ref_size = size, ref_size
        self.task = task
        if alpha_mode != "continuous" and any("alpha_mode" not in r for r in records):
            raise ValueError("alpha filtering requires a prepared catalog with inspected alpha_mode")
        selected = [r for r in records if r["split"] == split and
                    (asset_type == "all" or r["asset_type"] == asset_type) and
                    (task != "reference" or r.get("reference") is not None) and
                    (alpha_mode == "continuous" or r.get("alpha_mode") == "opaque" or
                     (alpha_mode == "binary" and r.get("alpha_mode") == "binary")) and
                    (not style or r.get("style") == style)]
        if target_scope == "paired":
            paired_hashes = {r["target_hash"] for r in selected if r.get("reference")}
            selected = [r for r in selected if r["target_hash"] in paired_hashes]
        # Text-only training must not double-weight targets just because they
        # have two generated references. Keep distinct captions for one target.
        if task == "text":
            unique = {}
            for r in selected:
                unique.setdefault((r["target_hash"], r["prompt"]), r)
            selected = list(unique.values())
        self.records = sorted(selected, key=lambda r: r["id"])
        self.maps = {}
        if not self.records:
            raise ValueError(f"empty {split} dataset for {task}/{asset_type}")
        groups = {True: defaultdict(list), False: defaultdict(list)}
        for i, r in enumerate(self.records):
            groups[task != "text" and r.get("reference") is not None][r["target_hash"]].append(i)
        self.groups = {k: list(v.values()) for k, v in groups.items()}

    def sample_indices(self, n, generator, paired_fraction):
        result = []
        for _ in range(n):
            paired = bool(torch.rand((), generator=generator) < paired_fraction)
            if not self.groups[paired]:
                paired = not paired
            groups = self.groups[paired]
            group = groups[int(torch.randint(len(groups), (), generator=generator))]
            result.append(group[int(torch.randint(len(group), (), generator=generator))])
        return result

    def __len__(self):
        return len(self.records)

    def array(self, spec):
        key = (spec["path"], tuple(spec["shape"]))
        if key not in self.maps:
            self.maps[key] = np.memmap(key[0], dtype=np.uint8, mode="r", shape=key[1])
        if not 0 <= spec["index"] < spec["shape"][0]:
            raise ValueError("mmap row outside shape")
        return np.array(self.maps[key][spec["index"]])

    def __getitem__(self, i):
        r = self.records[i]
        target = native_grid(self.array(r["target"]), self.size)
        if content_hash(target) != r["target_hash"]:
            raise ValueError(f"target hash mismatch for {r['id']}")
        ref = reference(self.array(r["reference"]), self.ref_size) if r.get("reference") and self.task != "text" else torch.zeros(4, self.ref_size, self.ref_size)
        return {"image": encode(target), "reference": ref, "ref_present": r.get("reference") is not None,
                "prompt": r["prompt"], "tileable": r["tileable"] is True, "id": r["id"]}

    def batch(self, indices, device):
        rows = [self[int(i)] for i in indices]
        return {"image": torch.stack([r["image"] for r in rows]).to(device),
                "reference": torch.stack([r["reference"] for r in rows]).to(device),
                "ref_present": torch.tensor([r["ref_present"] for r in rows], device=device),
                "tileable": torch.tensor([r["tileable"] for r in rows], device=device),
                "prompts": [r["prompt"] for r in rows], "ids": [r["id"] for r in rows]}
