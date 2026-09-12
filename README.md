# MC-Gen 2.0

Generate **Minecraft / voxel-game low-resolution textures** (32x32, RGBA) from
text with a compact **pixel-space Rectified-Flow Transformer**.

The model works directly on pixels (no VAE), is small enough to train on a
single 8 GB GPU, and produces seamless, palette-faithful textures for resource
packs, pixel-art assets and mod content.

- Model: `MC-FlowDiT` (MMDiT double-stream + FLUX-style single-stream, ~56 M params)
- Resolution: 32x32 x 4 (RGBA), patch size 2, 2D RoPE, RMSNorm + QK-Norm, AdaLN-Zero
- Objective: Rectified Flow / Flow Matching (pixel space) + optional tile loss
- Text conditioning: frozen pooled text embedding, precomputed offline
- License: **research / study only** (see *Licensing*)

> Datasets are hosted on Hugging Face: **https://huggingface.co/datasets/Risposta/MC_Gen**
> This repository contains the code only.

---

## Repository layout

```text
configs/
  data/      # dataset descriptors (stage_1, stage_2, ...)
  model/     # tiny / base / large / native16
  train/     # per-stage training recipes (stage_1.yaml, smoke_rgba.yaml, ...)
src/
  data/      # builders, loaders, processors
  model/     # MC-FlowDiT blocks
  train/     # flow, losses, EMA, trainer
  infer/     # solvers, sampling, FastAPI service
  eval/      # seam, attributes, diversity, memorization
scripts/     # entry points (train, sample, build_data, upload/download HF, smoke_test)
data/        # (downloaded, git-ignored) raw + built datasets
checkpoints/ # (git-ignored) model checkpoints
outputs/     # (git-ignored) logs and generated samples
```

---

## 1. Setup

```bash
git clone https://github.com/jurky123/MC_Gen2.0.git
cd MC_Gen2.0
pip install -r requirements.txt
```

Requirements: Python 3.11, PyTorch 2.x with CUDA, plus `transformers`,
`huggingface_hub[hf_xet]`, `pyarrow`, `pillow`, `numpy`, `fastapi`, `uvicorn`,
`imagehash`, `einops`.

---

## 2. Get the data

The training-ready datasets are on Hugging Face (`repo_type=dataset`):

```bash
pip install -U "huggingface_hub[hf_xet]"
python scripts/download_dataset_hf.py          # all datasets
python scripts/download_dataset_hf.py --only stage1_32_rgba   # just pretraining
```

This places files exactly where the configs expect them:

```text
data/build/stage1_32_rgba/            # Stage 1: pixel + all-MC pretraining (16 GB)
data/build/mc_text2image32/           # MC domain text+image
data/processed/minecraft_16x_finetune32/
data/processed/modrinth32/
```

To push updated datasets from a machine that produced them:

```bash
hf auth login
python scripts/upload_dataset_hf.py            # --dry-run to preview
```

See `hf/README.md` for the full data card.

---

## 3. Train

### Stage 1 — pixel + all-MC pretraining (ready)

```bash
scripts\train_stage1.cmd            # Windows
# or, on Linux/any OS:
python scripts/train.py --model configs/model/base.yaml --train configs/train/stage_1.yaml
```

Key recipe settings (`configs/train/stage_1.yaml`):

| setting | value |
|---|---|
| micro-batch | 16 |
| gradient accumulation | 8 (**effective batch 128**) |
| steps | 234,710 (~10 epochs) |
| optimizer | fused AdamW, lr 3e-4, betas (0.9, 0.95) |
| precision | bf16 + `torch.compile` |
| checkpoints | `checkpoints/stage_1/latest.pt` every 2,000 steps |

Stage 1 mixes several sources **at read time** (no merged copy and no re-upload
needed): `data/build/stage1_32_rgba` (mmap) plus `data/processed/modrinth32`
and `data/processed/minecraft_16x_finetune32` (tile manifests). Sources,
channels and optional sampling weights live in `configs/data/stage_1.yaml`;
Stage 1 uses images only.

Resume after interruption:

```bash
python scripts/train.py --model configs/model/base.yaml --train configs/train/stage_1.yaml \
  --resume checkpoints/stage_1/latest.pt
```

Fast sanity check (4-channel + accumulation + checkpointing, a few steps):

```bash
python scripts/train.py --model configs/model/base.yaml --train configs/train/smoke_rgba.yaml
```

### Other stages

- Stage 2 (MC weak labels) / Stage 3 (curated fine-tune) recipes live in
  `configs/train/stage_2.yaml`, `stage_c.yaml`, `stage_d.yaml` and their data
  descriptors in `configs/data/`. Stage 3 seeds are already uploaded.
- All training configs read `channels: 4` from their data YAML; the loader
  auto-adds an opaque alpha channel when a dataset is stored as RGB.

---

## 4. Inference

```bash
python scripts/sample.py \
  --ckpt checkpoints/stage_1/latest.pt \
  --prompt "dark mossy stone bricks" \
  --steps 20 --cfg 2.0 --solver heun
```

Or run the HTTP service:

```bash
set MC_CKPT=checkpoints/stage_1/latest.pt
python scripts/api.py     # POST /generate {prompt, seed, steps, cfg, solver}
```

Outputs are 32x32 RGBA PNGs plus a 4x4 tiled preview.

---

## 5. Data format

Built datasets are raw NumPy mmaps:

```text
<name>/
├── images.uint8.mmap   # headerless uint8, C-order (N, H, W, C)
├── metadata.parquet    # index, source_set, weak_prompt, source_metadata
├── splits.json         # {"train": [...], "val": [...], "test": [...]}
└── build_summary.json
```

Rebuild from processed manifests (requires the raw downloads):

```bash
python scripts/build_data.py stage1
```

---

## 6. Text encoder (frozen, offline)

Prompt conditioning uses a frozen encoder whose weights are never loaded during
MC-FlowDiT training. The single source of truth is `configs/text_encoder.yaml`
(`Qwen/Qwen3-VL-Embedding-2B`, single pooled + L2-normalized 2048-d token).

Precompute the per-sample embeddings used by Stage C/D:

```bash
python scripts/precompute_text.py --config configs/data/stage_c.yaml
```

This writes `text_embeddings.f32.mmap` (headerless fp32, shape `(N, 1, 2048)`)
next to the dataset. Keep `text_dim` / `max_text_tokens` in sync between
`configs/text_encoder.yaml`, the data config and `configs/model/base_qwen.yaml`
(Stage 1 uses the unconditional `base.yaml` and needs no text).

At inference the same encoder is selected with `--encoder-type qwen3vl`
(`--instruction` overrides the default prompt); the API accepts the matching
`encoder_type` / `instruction` fields.

---

## 7. Licensing

Research and study only. The datasets aggregate third-party game assets under
mixed licenses (CC0, CC-BY, OGA-BY and others); per-sample provenance is kept
in `source_metadata`. Minecraft / Mojang assets are **not** redistributed as
training data. Do not use the data or trained models commercially.
