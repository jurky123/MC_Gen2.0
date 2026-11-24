---
license: other
license_name: research-only
license_link: https://github.com/jurky123/MC_Gen2.0
pretty_name: "MC-Gen 2.0 — Minecraft item textures and generation checkpoints"
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

# MC Gen 2.0

用英文物品名、高清参考图，或两者一起，生成 **16×16 RGBA Minecraft 风格物品图**。本地编码器版权重为 **coarse-step8400**。提示词可以从 `poison apple`、`ghost boots`、`diamond sword` 这样的短名称开始。

[项目代码](https://github.com/jurky123/MC_Gen2.0) · [下载当前权重](https://huggingface.co/datasets/Risposta/MC_Gen/resolve/main/checkpoints/coarse-step8400/mcgen-coarse-step8400.pt)

| 版本 | 条件编码方式 | 使用场景 |
| --- | --- | --- |
| [本地版 · coarse-step8400](https://huggingface.co/datasets/Risposta/MC_Gen/tree/main/checkpoints/coarse-step8400) | 本机 Qwen3-VL-Embedding-8B 提取文本和视觉 token | 离线生成 |
| [API 默认版 · api-step10516](https://huggingface.co/datasets/Risposta/MC_Gen/tree/main/checkpoints/api-step10516) | 北京百炼 `qwen3-vl-embedding` 分别返回文字和参考图的 2560 维向量 | 本机仅运行 MC 生成器 |

[API 版使用说明](https://huggingface.co/datasets/Risposta/MC_Gen/tree/main/checkpoints/api-step10516) · [GitHub 快速开始](https://github.com/jurky123/MC_Gen2.0#-快速开始)

## 生成效果

![文件名粗标签生成的 16 个物品](https://huggingface.co/datasets/Risposta/MC_Gen/resolve/main/checkpoints/coarse-step8400/text.png)

上图取自广域验证抽样的前 16 个提示，全部使用同一种子。下图来自高清配对测试集，每行从左到右为高清参考、MC 目标、纯文字、文字加参考、纯参考。

![同一测试物品在三种输入下的生成结果](https://huggingface.co/datasets/Risposta/MC_Gen/resolve/main/checkpoints/coarse-step8400/reference.png)

在 263 个配对测试目标上，图文联合的轮廓 IoU 为 **0.661**，纯参考图为 **0.639**，纯文字为 **0.556**。短名称往往对应多种合理外观，这组数字衡量与配对目标的接近程度。

## 本地编码器版快速使用

在项目根目录安装依赖并下载生成器及冻结编码器。

```bash
pip install -e '.[embedding]'
curl -fL --create-dirs -o checkpoints/mcgen-coarse-step8400.pt \
  https://huggingface.co/datasets/Risposta/MC_Gen/resolve/main/checkpoints/coarse-step8400/mcgen-coarse-step8400.pt
python scripts/download_qwen_vl_embedding.py --out ../Qwen3-VL-Embedding-8B
export MCGEN_EMBEDDING_MODEL=../Qwen3-VL-Embedding-8B
mcgen sample --checkpoint checkpoints/mcgen-coarse-step8400.pt \
  --prompt 'diamond sword' --steps 32 --text-scale 2 --seed 42 \
  --device cuda:0 --out outputs/sword.png
```

图文联合生成时添加 `--reference item.png --conditioning-mode visual_tokens`。纯参考图生成时省略 `--prompt`。参考图测试采用 `--text-scale 1 --ref-scale 1`。

[权重及校验文件](https://huggingface.co/datasets/Risposta/MC_Gen/tree/main/checkpoints/coarse-step8400) · [冻结编码器](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B)

## 训练数据

从零训练生成器，使用 284,825 条广域文件名标签，以及 3,586 个高清与 MC 配对训练目标。配对中 2,472 个使用来源文件名提取的标签，其余 1,114 个使用数据包中的简短物品名。先训练文生图，再从第 5,500 步底座学习高清参考条件，最后混合训练三种生成模式。

> **License: research / study only.** The datasets aggregate third-party game
> assets with heterogeneous licenses. They are published here for academic and
> personal study only. Do not use them commercially, and keep the provenance
> fields in the metadata for attribution. See *Provenance* below.

## 最新 MC/HD 配对数据

| 发布 | 配对数 |
| --- | --- |
| 通用合集 HDRepair v5 | 3,592 主集 / 3,685 含历史试验 |
| 变体矩阵 HDRepair v3 | 970（198 首批 + 772 追加） |

数据用途、训练索引、修复与隔离统计、下载链接和 SHA-256 见[配对数据发布说明](releases/README.md)。两份 ZIP 独立发布，跨包合并需去重。

<!-- reference-data-20261107:start -->
## Reference image pairs · 2026-11-07

[Data and labels](data/reference-20261107/README.md) · [Files](https://huggingface.co/datasets/Risposta/MC_Gen/tree/main/data/reference-20261107)

1,000 user-curated high-quality HD/MC pairs, plus reviewed legacy pairs. Captions, original filenames, provenance and existing splits are preserved.
The current **lossless16** version has **1,064 training / 1,350 validation / 56 test pairs** (864 / 711 / 56 distinct targets).
The other 500 genuine32×32 targets are archived separately and excluded from16×16 training; no lossy32→16 candidates are included.
Download the high-quality-native16 and reviewed-legacy archives for the current training version. See the data card for extraction and per-source attribution.

![Reference data and actual training labels](data/reference-20261107/preview.png)
<!-- reference-data-20261107:end -->

## Contents

| Path | Rows | Shape / dtype | Split (train/val/test) | Purpose |
|---|---|---|---|---|
| `data/build/stage1_32_rgba` | 3,185,719 | `uint8[32,32,4]` RGBA | 3,004,255 / 91,566 / 89,898 | **Stage 1** pixel + all-MC pretraining |
| `data/build/mc_text2image32` | 1,034,057 | `uint8[32,32,3]` RGB | 941,104 / 43,184 / 49,769 | Minecraft domain (text+image), legacy RGB build |
| `data/build/mc_text2image32_wl` | 1,034,057 | `uint8[32,32,4]` RGBA | 941,104 / 43,184 / 49,769 | **Stage 1/2 main MC set**: RGBA + weak/fine prompts + annotations + frozen-text embeddings |
| `data/build/stage2_32` | 1,044,875 | `uint8[32,32,4]` RGBA | 949,165 / 44,417 / 51,293 | **Stage 2** unified texture set (MC-wl + Modrinth) + pooled text embeddings |
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

### Extra files in `mc_text2image32_wl` / `stage2_32`

| file | meaning |
|---|---|
| `text_embeddings.f32.mmap` | frozen Qwen3-8B text features, FP32; `mc_text2image32_wl` = 4 prompt views × 4096-d per row, `stage2_32` = 1 pooled 4096-d token per row. **Note:** embeddings are currently *not* in this repo (too large to push); recompute with `scripts/precompute_text.py` in the project repo. |
| `prompt_views.*` / `grounded_prompts.*` / `filename_prompts.parquet` | prompt variants (K-view, label-grounded rewrite, filename weak prompts) |
| `coarse_annotations.shard*.jsonl` | local-27B VLM captions (coarse profile) |
| `annotation_manifest.jsonl` / `high_quality_manifest.jsonl` | annotation lineage (`stage2_32`) |
| `replay_splits*.json` | anti-forgetting replay index sets |

`metadata.parquet` columns for the unified datasets (`stage1_32_rgba`,
`stage2_32`):

| column | meaning |
|---|---|
| `index` | row index into the mmap |
| `source_set` | originating processed set (see table above) |
| `weak_prompt` | short text caption (MC filename/label or generic pixel-art) |
| `source_metadata` | JSON blob with original filename, project, license, etc. |

`mc_text2image32_wl/metadata.parquet` additionally carries Modrinth provenance
(`file_name`, `project_id`, `mod_slug`, `author`, `license`, `version_url`,
`type`, `tileable`) and the `text` / `weak_prompt` caption pair.

The processed `*32` tile folders instead contain `tiles/<sha>.png` plus a
`manifest.jsonl` and `summary.json`.

## File format

Each mmap dataset directory contains at least:

```text
<name>/
├── images.uint8.mmap   # raw little-endian uint8, C-order (N, H, W, C)
├── metadata.parquet    # one row per image + provenance/prompt fields
├── splits.json         # {"train": [idx...], "val": [...], "test": [...]}
└── build_summary.json  # counts and source breakdown (where present)
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

## Download

```bash
pip install -U "huggingface_hub[hf_xet]"

# only the pretraining set
hf download Risposta/MC_Gen \
  --repo-type dataset \
  --include "data/build/stage1_32_rgba/*" \
  --local-dir .

# everything (all datasets)
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
