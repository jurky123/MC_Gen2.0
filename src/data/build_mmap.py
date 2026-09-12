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
        channels=3,
    ):
        self.image_size = image_size
        self.toroidal = toroidal
        self.normalize = normalize
        self.text_dim = text_dim
        self.max_text_tokens = max_text_tokens
        self.channels = channels

        with open(splits, "r") as f:
            all_splits = json.load(f)
        self.index = all_splits.get(split, [])
        self.index = list(self.index)

        import pandas as pd

        self.df = pd.read_parquet(metadata)
        self.n = len(self.df)
        self.disk_channels = self._infer_disk_channels(images, image_size)
        self.images = np.memmap(
            images, dtype=np.uint8, mode="r",
            shape=(self.n, image_size, image_size, self.disk_channels),
        )
        self.has_text = bool(text_mmap) and os.path.exists(text_mmap)
        if self.has_text:
            self.text = np.memmap(text_mmap, dtype=np.float32, mode="r", shape=(self.n, max_text_tokens, text_dim))
        self.df = self.df

    def _infer_disk_channels(self, images, image_size):
        with open(images, "rb") as handle:
            if handle.read(6) == b"\x93NUMPY":
                return int(np.load(images, mmap_mode="r").shape[-1])
        total = os.path.getsize(images)
        for c in (self.channels, 4, 3, 1):
            if total == self.n * image_size * image_size * c:
                return c
        raise ValueError(
            f"cannot infer channel count for {images}: {total} bytes, {self.n} rows at {image_size}px"
        )

    def __len__(self):
        return len(self.index)

    def _adapt_channels(self, arr):
        disk = arr.shape[-1]
        if disk == self.channels:
            return arr
        if disk == 3 and self.channels == 4:
            alpha = np.full(arr.shape[:2] + (1,), 255, dtype=np.uint8)
            return np.concatenate([arr, alpha], axis=-1)
        if disk == 4 and self.channels == 3:
            rgb = arr[..., :3].astype(np.uint16)
            a = arr[..., 3:4].astype(np.uint16)
            return ((rgb * a + 127) // 255).astype(np.uint8)
        raise ValueError(f"cannot adapt {disk}-channel image to {self.channels} channels")

    def __getitem__(self, i):
        idx = self.index[i]
        arr = self._adapt_channels(np.asarray(self.images[idx]))
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


class ManifestImageDataset(torch.utils.data.Dataset):
    """In-memory dataset over a processed tile manifest (tiles/ + manifest.jsonl).

    Tiles are decoded once at construction, so this is only suitable for the
    small processed sets (a few thousand images).
    """

    _SPLITS = ("train", "val", "test")

    def __init__(self, manifest, image_size=32, channels=3, normalize=True, split="train",
                 text_dim=768, toroidal=False, val_fraction=0.05, test_fraction=0.05):
        base = Path(__file__).resolve().parents[2]
        self.image_size = image_size
        self.channels = channels
        self.normalize = normalize
        self.text_dim = text_dim
        self.toroidal = toroidal

        records = []
        with Path(manifest).open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        images = np.zeros((len(records), image_size, image_size, channels), dtype=np.uint8)
        index = {name: [] for name in self._SPLITS}
        train_threshold = 100.0 * (1.0 - val_fraction - test_fraction)
        val_threshold = 100.0 * (1.0 - test_fraction)
        for i, rec in enumerate(records):
            rel = str(rec["path"]).replace("\\", "/")
            path = Path(rel)
            if not path.is_absolute():
                path = base / rel
            image = Image.open(path).convert("RGBA")
            if image.size != (image_size, image_size):
                image = image.resize((image_size, image_size), Image.Resampling.NEAREST)
            if channels == 4:
                images[i] = np.asarray(image, dtype=np.uint8)
            else:
                background = Image.new("RGBA", image.size, (0, 0, 0, 255))
                background.alpha_composite(image)
                images[i] = np.asarray(background.convert("RGB"), dtype=np.uint8)
            name = str(rec.get("split") or "")
            if name == "validation":
                name = "val"
            if name not in self._SPLITS:
                digest = Path(rel).stem
                value = int(hashlib.sha1(digest.encode("utf-8")).hexdigest()[:8], 16) % 100
                name = "train" if value < train_threshold else (
                    "val" if value < val_threshold else "test")
            index[name].append(i)
        self.images = images
        self.index = list(index.get(split, []))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        arr = self.images[self.index[i]]
        if self.toroidal:
            arr = np.roll(arr, (np.random.randint(0, self.image_size),
                               np.random.randint(0, self.image_size)), axis=(0, 1))
        x = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).float() / 127.5 - 1.0
        emb = torch.zeros(1, self.text_dim, dtype=torch.float32)
        return x, emb


class MixDataset(torch.utils.data.Dataset):
    """Concatenate several datasets; ``make_sampler`` enables weighted sampling."""

    def __init__(self, datasets, weights=None):
        self.datasets = list(datasets)
        self.weights = [float(w) for w in (weights or [1.0] * len(self.datasets))]
        if len(self.weights) != len(self.datasets):
            raise ValueError("weights length must match datasets")
        self._ends = []
        total = 0
        for dataset in self.datasets:
            total += len(dataset)
            self._ends.append(total)

    def __len__(self):
        return self._ends[-1] if self._ends else 0

    def __getitem__(self, i):
        if i < 0:
            i += len(self)
        for k, end in enumerate(self._ends):
            if i < end:
                start = self._ends[k - 1] if k else 0
                return self.datasets[k][i - start]
        raise IndexError(i)

    def make_sampler(self, num_samples=None, generator=None):
        if all(abs(w - 1.0) < 1e-9 for w in self.weights):
            return None
        items = []
        for dataset, weight in zip(self.datasets, self.weights):
            n = len(dataset)
            if n:
                items.append(torch.full((n,), weight / n, dtype=torch.double))
        if not items:
            return None
        return torch.utils.data.WeightedRandomSampler(
            torch.cat(items), num_samples or len(self), replacement=True, generator=generator)