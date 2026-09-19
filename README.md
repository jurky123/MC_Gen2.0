# MC-Gen 2.0

Generate **low-resolution voxel-game / pixel-art textures** (32×32, RGBA) from text with a
compact **pixel-space Rectified-Flow Transformer**.

The model works directly on pixels (no VAE), trains on a single GPU, and targets seamless,
palette-faithful textures.

- Model: `MC-FlowDiT` (~130 M) — pixel-space MMDiT double-stream + single-stream
- Resolution: 32×32 × 4 (RGBA), patch size 2, 2D RoPE, RMSNorm + QK-Norm, AdaLN-Zero
- Objective: Rectified Flow / Flow Matching + optional edge (tile) loss
- Text conditioning: frozen **Qwen3-8B** token states injected by **cross-attention**
- License: **research / study only**

---

## Results

Selected generations from the current checkpoint (nearest-neighbour upscaled;
transparent background shown white):

![generations](docs/images/generated.png)

---

## Features

- **Pixel-space flow model** — no VAE, so pixel edges, palette and tile seams are preserved.
- **Cross-attention text conditioning** — image tokens query a frozen Qwen3-8B token
  sequence (hidden states from layers 9/18/27 concatenated → 12 288-d), computed **on the
  fly** (no precomputed text embeddings; the text tower runs on a second GPU).
- **Label-grounded prompts** — the source label words are authoritative: orientation and
  abstract words are preserved verbatim, the `block` / `item` type is kept, and colour /
  shape are inferred from the label; an optional LLM rewrite keeps every label word.
- **Curated fine-tuning set** — a category-balanced subset (63 forms × 61 materials × 30
  colours × 48 states) annotated by a local 27B VLM under a strict, validated JSON rule
  (15 visual fields, verbatim label preservation, fallback on validation failure).
- **Anti-forgetting fine-tuning** — fine-tuning mixes in replay data, uses a low LR and a
  short schedule, CPU EMA, an optional backbone freeze (train only the conditioning path)
  and forgetting-aware checkpoint selection.
- **Evaluation suite** — concept recall, colour accuracy, real-vs-generated fidelity, seam
  score, seed diversity, retrieval accuracy and text-effect (val loss vs null).

---

## Method (brief)

```text
Stage 1  pixel-art pretraining (image-only)          → pixel flow prior
   ↓
Stage 2  broad texture set + grounded prompts        → text alignment
   ↓
Stage 3  curated subset + 27B fine annotations
         + replay mixing                              → precise text control
```

**Prompt pipeline** — each texture keeps its source-label words. A deterministic grounded
rewrite (`scripts/rewrite_grounded.py`) and a validated LLM rewrite
(`src/data/grounded_rewrite.py`) both must keep every label word and the asset type.

**Text tower** — `Qwen3-8B` causal LM, chat template with `enable_thinking=False`, hidden
states from layers 9/18/27 concatenated (FLUX.2-klein style); frozen, run on a second GPU.
See `src/data/text_tower.py`.

**Model** — `src/model/mc_flow_dit.py`: patch embed → MMDiT blocks → cross-attention to text
tokens → single-stream blocks → velocity head, with 2D RoPE, QK-Norm and AdaLN-Zero.
`text_injection: cross_attn` keeps a learned register token so Stage-1 weights transfer.

**Inference** — Heun / Euler flow ODE, 16–24 steps, CFG ≈ 2–3, frozen text tower with a
learned null condition.

---

## Repository layout

```text
configs/
  data/      dataset descriptors (stage_1, stage_2_grounded_k1, stage_3, ...)
  model/     tiny / base / base_flux2klein
  train/     per-stage recipes (stage_1, stage_2_grounded_k1, stage_3, stage_3_frozen)
  annotator.yaml, prompt_rewrite.yaml, grounded_rewrite.yaml, stage3_annotator.yaml
src/
  data/      builders, loaders, text tower, annotators, prompt pipelines
  model/     MC-FlowDiT blocks (double/single stream, cross-attention, RoPE)
  train/     flow, losses, EMA, trainer
  infer/     solvers, sampling, FastAPI service
  eval/      seam, attributes, diversity, memorization, caption bench
scripts/     train / sample / annotate / rewrite / eval entry points
docs/images/ result figures
data/        (git-ignored) raw and built datasets
```

---

## Quickstart

```bash
pip install -r requirements.txt

# Stage 2 — text alignment (Qwen3-8B cross-attention)
python scripts/train.py \
  --model configs/model/base_flux2klein.yaml \
  --train configs/train/stage_2_grounded_k1.yaml \
  --init-from checkpoints/stage_1/latest.pt

# Stage 3 — annotated subset + replay, anti-forgetting
python scripts/train.py \
  --model configs/model/base_flux2klein.yaml \
  --train configs/train/stage_3.yaml \
  --init-from checkpoints/stage_2_grounded_k1/best.pt
# optional frozen-conditioning variant: configs/train/stage_3_frozen.yaml

# sample
python scripts/sample.py \
  --ckpt checkpoints/stage_3/best.pt \
  --text-tower /path/to/Qwen3-8B --text-layers 9,18,27 \
  --prompt "orange brick item" "mossy stone bricks block" \
  --steps 20 --cfg 2.5 --solver heun

# evaluate a checkpoint
python scripts/eval_generation.py --ckpt checkpoints/stage_3/best.pt
```

Serving a local annotator with vLLM (one replica per GPU):

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/serve_vlm.py \
  --model /path/to/Qwen3.8-27B --served-model-name qwen3.8-27b --port 8000 --mtp 2
```

---

## License

Research and study only. Do not use the trained models commercially.
