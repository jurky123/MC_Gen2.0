---
license: other
license_name: research-only
license_link: https://github.com/jurky123/MC_Gen2.0
pretty_name: "MC-Gen 2.0 — Minecraft / pixel-art 32x32 texture datasets"
task_categories:
  - text-to-image
tags:
  - minecraft
  - pixel-art
  - texture
  - diffusion
  - flow-matching
  - mmap
  - research
size_categories:
  - 1M<n<10M
---

# MC-Gen 2.0 — training datasets

Training-ready datasets for the **MC-Gen 2.0** project: a small pixel-space
Rectified-Flow Transformer that generates Minecraft / voxel-game style
low-resolution textures (32x32, RGBA).

The source code lives on GitHub: **https://github.com/jurky123/MC_Gen2.0**

This repository only contains the **preprocessed, training-ready** datasets.
Raw downloads (itch.io, OpenGameArt, Kaggle, Modrinth, Hugging Face mirrors)
are intentionally not included.

> **License: research / study only.** The datasets aggregate third-party game
> assets with heterogeneous licenses. They are published here for academic and
> personal study only. Do not use them commercially, and keep the provenance
> fields in the metadata for attribution. See *Provenance* below.

## Contents

| Path | Rows | Shape / dtype | Split (train/val/test) | Purpose |
|---|---|---|---|---|
| `data/build/stage1_32_rgba` | 3,185,719 | `uint8[32,32,4]` RGBA | 3,004,255 / 91,566 / 89,898 | **Stage 1** pixel + all-MC pretraining |
| `data/build/mc_text2image32` | 1,034,057 | `uint8[32,32,3]` RGB | 941,104 / 43,184 / 49,769 | Minecraft domain (text+image), Stage 1/2 |
| `data/processed/minecraft_16x_finetune32` | 1,498 | `32x32` tiles | manifest only | **Stage 3** curated fine-tune seeds |
| `data/processed/modrinth32` | 10,818 | `32x32` tiles | manifest only | Modrinth MC textures (deduped) |

### `stage1_32_rgba` source mix

| source_set | rows |
|---|---:|
| itch.io free pixel art | 1,859,029 |
| NathMen12 MC text-to-image | 1,034,057 |
| alucard pixel sprites | 282,044 |
| OpenGameArt (OGA-BY 3.0) | 4,555 |
| Kenney pixel assets | 4,312 |
| Kaggle pixel art | 1,722 |

Minecraft rows are fully opaque; pixel-art rows keep their original alpha
channel (transparent backgrounds are **not** composited onto black).

## File format

Each mmap dataset directory contains:

```text
<name>/
├── images.uint8.mmap   # raw little-endian uint8, C-order (N, H, W, C)
├── metadata.parquet    # one row per image + provenance/prompt fields
├── splits.json         # {"train": [idx...], "val": [...], "test": [...]}
└── build_summary.json  # counts and source breakdown
```

`images.uint8.mmap` is a headerless NumPy array. Load it with:

```python
import numpy as np
n, size, channels = 3_185_719, 32, 4
imgs = np.memmap(
    "data/build/stage1_32_rgba/images.uint8.mmap",
    dtype=np.uint8, mode="r", shape=(n, size, size, channels),
)
```

`metadata.parquet` columns for the unified datasets:

| column | meaning |
|---|---|
| `index` | row index into the mmap |
| `source_set` | originating processed set (see table above) |
| `weak_prompt` | short text caption (MC filename/label or generic pixel-art) |
| `source_metadata` | JSON blob with original filename, project, license, etc. |

The processed `*32` tile folders instead contain `tiles/<sha>.png` plus a
`manifest.jsonl` and `summary.json`.

## Download

```bash
pip install -U "huggingface_hub[hf_xet]"

# only the pretraining set
hf download Risposta/MC_Gen \
  --repo-type dataset \
  --include "data/build/stage1_32_rgba/*" \
  --local-dir .

# everything (all four datasets)
hf download Risposta/MC_Gen --repo-type dataset --local-dir .
```

With `--local-dir .` run from the project root, the files land exactly at the
paths the training configs expect. A convenience wrapper is provided in the
project repo as `scripts/download_dataset_hf.py`.

## Provenance

The datasets derive from the following sources (see `source_metadata` for
per-sample details): NathMen12/16xModdedMinecraft-TextToImage, itch.io free
pixel-art asset packs, Kenney CC0 packs, evilsocket/alucard-sprites, Kaggle
ebrahimelgazar/pixel-art, nyuuzyou/OpenGameArt-OGA-BY-3.0,
James-A/Minecraft-16x-Dataset, and Modrinth resource packs.

This repository is **not affiliated with, endorsed by, or associated with**
Microsoft, Mojang, or any asset author. All trademarks belong to their
respective owners.

## Citation

```bibtex
@misc{mcgen2,
  title  = {MC-Gen 2.0: pixel-space flow transformer for Minecraft textures},
  author = {Risposta},
  year   = {2026},
  url    = {https://github.com/jurky123/MC_Gen2.0}
}
```
