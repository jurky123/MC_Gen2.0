import json
import hashlib
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from data import mmap_io
from data.rgba import premultiply_rgba_np


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


def build_mmap(out_dir, records, image_size=32, split_by="project_id", splits=None, seed=0,
               channels=4, group_split_seed=None, fracs=(0.9, 0.05, 0.05)):
    """Build a headerless images.uint8.mmap + metadata.parquet + splits.json.

    ``split_by`` names metadata column(s) whose joint value defines a split
    group (e.g. project_id): every row of the same group lands in the same
    split (P0-2). Splits are deterministic given the seed. An explicitly
    passed ``splits`` mapping is honoured (validated); otherwise a group
    split is generated. ``seed`` is the canonical split seed
    (``group_split_seed`` is a deprecated alias, P1-5).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(records)
    images = np.empty((n, image_size, image_size, channels), dtype=np.uint8)
    for i, rec in enumerate(records):
        img = Image.open(rec["path"]).convert("RGBA")
        tile = img.resize((image_size, image_size), Image.Resampling.NEAREST)
        if channels == 4:
            images[i] = np.asarray(tile, dtype=np.uint8)
        else:
            bg = Image.new("RGBA", tile.size, (0, 0, 0, 255))
            bg.alpha_composite(tile)
            images[i] = np.asarray(bg.convert("RGB"), dtype=np.uint8)
        if (i + 1) % 2000 == 0:
            print(f"  wrote {i + 1}/{n}")
    mmap_io.write_raw_mmap(images, out_dir / "images.uint8.mmap")
    mmap_io.write_schema(out_dir, "images.uint8.mmap", images.shape, np.uint8,
                         extra={"image_size": image_size, "channels": channels})

    import pandas as pd

    df = pd.DataFrame.from_records(records)
    df = df.drop(columns=["path"], errors="ignore")
    df.to_parquet(out_dir / "metadata.parquet", index=False)

    from data.lineage_split import check_disjoint, group_split

    if group_split_seed is None:
        group_split_seed = seed
    if splits is None:
        roles, audit = group_split(df.to_dict("records"), group_keys=split_by,
                                   seed=group_split_seed, fracs=fracs)
        splits = {role: [i for i, r in sorted(roles.items()) if r == role]
                  for role in ("train", "val", "test")}
        (out_dir / "split_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
        print(f"splits (group={split_by}): {audit['split_sizes']} groups={audit['n_groups']}")
    else:
        # Honour caller-provided splits, but validate them (P1-5).
        check_disjoint({k: set(v) for k, v in splits.items()}, "explicit splits")
        covered = sorted(i for v in splits.values() for i in v)
        assert covered == list(range(n)), "explicit splits must cover all rows exactly once"
        print(f"splits: explicit, sizes={ {k: len(v) for k, v in splits.items()} }")
    with open(out_dir / "splits.json", "w") as f:
        json.dump(splits, f)
    return out_dir


def write_text_mmap(embeddings, out_path, out_dir=None, name=None):
    # Headerless raw dump so MmapImageTextDataset can read it with np.memmap(shape=...).
    arr = np.ascontiguousarray(embeddings, dtype=np.float32)
    mmap_io.write_raw_mmap(arr, out_path)
    if out_dir is not None:
        mmap_io.write_schema(out_dir, name or Path(out_path).name, arr.shape, np.float32)


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
        text_views=0,
        prompt_views="",
        prompt_cols=None,
        toroidal_col="",
        toroidal_values=None,
        rgba_mode="straight",
        return_aux=False,
        tileable_col="tileable",
    ):
        assert rgba_mode in ("straight", "premultiplied")
        self.image_size = image_size
        self.toroidal = toroidal
        self.normalize = normalize
        self.text_dim = text_dim
        self.max_text_tokens = max_text_tokens
        self.channels = channels
        self.rgba_mode = rgba_mode
        self.return_aux = return_aux
        # Number of pooled conditioning views per sample (0 = legacy: use
        # max_text_tokens as the middle mmap dim). When > 1 the loader returns
        # one randomly chosen view so the same image sees different prompts.
        self.text_views = int(text_views) if text_views else 0

        with open(splits, "r") as f:
            all_splits = json.load(f)
        self.index = all_splits.get(split, [])
        self.index = list(self.index)

        import pandas as pd

        self.df = pd.read_parquet(metadata)
        self.n = len(self.df)
        # Toroidal roll augmentation is only valid for tileable textures; apply
        # it to rows selected by (toroidal_col in toroidal_values). Empty values
        # => every row (legacy behaviour).
        self.tileable = None
        if toroidal and toroidal_col and toroidal_col in self.df.columns:
            vals = list(toroidal_values or [])
            self.tileable = self.df[toroidal_col].astype(str).isin(vals).to_numpy()
        # Per-sample tileability used for masked seam/tile loss. Explicit
        # metadata column wins; otherwise fall back to asset_type heuristic.
        self.sample_tileable = None
        if tileable_col and tileable_col in self.df.columns:
            self.sample_tileable = self.df[tileable_col].astype(str).str.lower().isin(
                ("true", "1", "yes")).to_numpy()
        elif "type" in self.df.columns:
            self.sample_tileable = (self.df["type"].astype(str) == "block").to_numpy()
        self.disk_channels = self._infer_disk_channels(images, image_size)
        self.images = np.memmap(
            images, dtype=np.uint8, mode="r",
            shape=(self.n, image_size, image_size, self.disk_channels),
        )
        self.has_text = bool(text_mmap) and os.path.exists(text_mmap)
        self.text_middle = self.text_views if self.text_views > 0 else max_text_tokens
        if self.has_text:
            self.text = np.memmap(
                text_mmap, dtype=np.float32, mode="r",
                shape=(self.n, self.text_middle, text_dim),
            )

        # Dynamic (on-the-fly) prompt strings for cross-attention conditioning.
        # When set, __getitem__ returns a text string instead of a text embedding.
        self.prompt_cols = list(prompt_cols or [])
        self.has_prompts = bool(prompt_views) and os.path.exists(prompt_views) and bool(self.prompt_cols)
        if self.has_prompts:
            import pandas as pd

            pv = pd.read_parquet(prompt_views)
            missing = [c for c in self.prompt_cols if c not in pv.columns]
            if missing:
                raise ValueError(f"prompt_views missing columns: {missing}")
            self.prompts = pv[self.prompt_cols].fillna("").astype(str).to_numpy()
            if len(self.prompts) != self.n:
                raise ValueError(f"prompt_views has {len(self.prompts)} rows, dataset has {self.n}")

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
        if self.rgba_mode == "premultiplied" and arr.shape[-1] == 4:
            arr = premultiply_rgba_np(arr)
        dy = dx = 0
        do_roll = self.toroidal and (self.tileable is None or bool(self.tileable[idx]))
        if do_roll:
            dy = np.random.randint(0, self.image_size)
            dx = np.random.randint(0, self.image_size)
            arr = np.roll(arr, (dy, dx), axis=(0, 1))
        x = torch.from_numpy(arr).permute(2, 0, 1).contiguous().float() / 127.5 - 1.0
        if self.has_prompts:
            view = np.random.randint(0, len(self.prompt_cols))
            text = str(self.prompts[idx, view])
        elif self.has_text:
            if self.text_views > 1:
                view = np.random.randint(0, self.text_views)
                emb = torch.from_numpy(np.asarray(self.text[idx, view]).copy()).float()[None, :]
            else:
                emb = torch.from_numpy(np.asarray(self.text[idx]).copy()).float()
            text = emb
        else:
            # Unconditional: a single null token (must match inference, which
            # also feeds one null token for the placeholder/hash encoder).
            text = torch.zeros(1, self.text_dim, dtype=torch.float32)
        if not self.return_aux:
            return x, text
        aux = {
            "tileable": torch.tensor(
                bool(self.sample_tileable[idx]) if self.sample_tileable is not None else False,
                dtype=torch.bool),
            "asset_type": torch.tensor(
                1 if str(self.df.iloc[idx].get("type")) == "item" else 0, dtype=torch.long),
        }
        return x, text, aux


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