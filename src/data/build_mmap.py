import json
import hashlib
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _crop_resize(image, crop_size, target_size, rng):
    w, h = image.size
    if crop_size >= min(w, h):
        box = (0, 0, w, h)
    else:
        x = rng.randint(0, w - crop_size)
        y = rng.randint(0, h - crop_size)
        box = (x, y, x + crop_size, y + crop_size)
    return image.crop(box).resize((target_size, target_size), Image.Resampling.BICUBIC)


def _to_rgb_uint8(image, target_size):
    return image.convert("RGB").resize((target_size, target_size), Image.Resampling.NEAREST)


def build_mmap(out_dir, records, image_size=32, split_by="project_id", splits=None, seed=0):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(records)
    images = np.lib.format.open_memmap(
        str(out_dir / "images.uint8.mmap"), mode="w+", dtype=np.uint8, shape=(n, image_size, image_size, 3)
    )
    for i, rec in enumerate(records):
        img = Image.open(rec["path"]).convert("RGB")
        images[i] = np.asarray(_to_rgb_uint8(img, image_size), dtype=np.uint8)
        if (i + 1) % 2000 == 0:
            print(f"  wrote {i + 1}/{n}")
    images.flush()
    del images

    import pandas as pd

    df = pd.DataFrame.from_records(records)
    df = df.drop(columns=["path"], errors="ignore")
    df.to_parquet(out_dir / "metadata.parquet", index=False)

    ids = list(df.index)
    if splits is None:
        rng = np.random.RandomState(seed)
        ids = rng.permutation(ids)
        n_train = int(len(ids) * 0.9)
        n_val = int(len(ids) * 0.05)
        splits = {
            "train": [int(i) for i in ids[:n_train]],
            "val": [int(i) for i in ids[n_train : n_train + n_val]],
            "test": [int(i) for i in ids[n_train + n_val :]],
        }
    with open(out_dir / "splits.json", "w") as f:
        json.dump(splits, f)
    return out_dir


def write_text_mmap(embeddings, out_path):
    arr = np.asarray(embeddings, dtype=np.float32)
    mem = np.lib.format.open_memmap(str(out_path), mode="w+", dtype=np.float32, shape=arr.shape)
    mem[:] = arr
    mem.flush()
    del mem


class MmapImageTextDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        images,
        metadata,
        splits,
        split="train",
        image_size=32,
        toroidal=True,
        normalize=True,
        text_mmap="",
        text_dim=768,
        max_text_tokens=64,
    ):
        self.image_size = image_size
        self.toroidal = toroidal
        self.normalize = normalize
        self.text_dim = text_dim
        self.max_text_tokens = max_text_tokens

        with open(splits, "r") as f:
            all_splits = json.load(f)
        self.index = all_splits.get(split, [])
        self.index = list(self.index)

        import pandas as pd

        self.df = pd.read_parquet(metadata)
        self.n = len(self.df)
        self.images = np.memmap(images, dtype=np.uint8, mode="r", shape=(self.n, image_size, image_size, 3))
        self.has_text = bool(text_mmap) and os.path.exists(text_mmap)
        if self.has_text:
            self.text = np.memmap(text_mmap, dtype=np.float32, mode="r", shape=(self.n, max_text_tokens, text_dim))
        self.df = self.df

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        idx = self.index[i]
        arr = np.asarray(self.images[idx])
        dy = dx = 0
        if self.toroidal:
            dy = np.random.randint(0, self.image_size)
            dx = np.random.randint(0, self.image_size)
            arr = np.roll(arr, (dy, dx), axis=(0, 1))
        x = torch.from_numpy(arr).permute(2, 0, 1).contiguous().float() / 127.5 - 1.0
        if self.has_text:
            emb = torch.from_numpy(np.asarray(self.text[idx]).copy()).float()
        else:
            emb = torch.zeros(1, self.text_dim, dtype=torch.float32)
        return x, emb