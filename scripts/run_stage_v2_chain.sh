#!/bin/bash
# Chain: wait for Stage-2 v2 -> Stage-3 v2 (frozen, leak-free splits) -> evals.
set -x
cd /home/iflab/models/MC_Gen2.0
PY=/home/iflab/miniconda3/envs/mcgen/bin/python

# wait for the running Stage-2 v2 training to finish
while pgrep -f "[s]cripts/train.py" > /dev/null; do sleep 60; done
sleep 5

# Stage 3 v2 (frozen) from Stage-2 v2 best
$PY scripts/train.py \
  --model configs/model/base_flux2klein.yaml \
  --train configs/train/stage_3_frozen_v2.yaml \
  --init-from checkpoints/stage_2_grounded_k1_v2/best.pt || exit 1

# Evals (same seed inside eval_generation.py; fixed splits/prompts per run)
$PY scripts/eval_generation.py \
  --ckpt checkpoints/stage_2_grounded_k1_v2/best.pt \
  --out outputs/eval_v2_stage2 || exit 1
$PY scripts/eval_generation.py \
  --ckpt checkpoints/stage_3_frozen_v2/best.pt \
  --out outputs/eval_v2_stage3_forget || exit 1
$PY scripts/eval_generation.py \
  --ckpt checkpoints/stage_3_frozen_v2/best.pt \
  --splits data/build/mc_text2image32_wl/stage3_splits.json \
  --prompts data/build/mc_text2image32_wl/stage3_prompts.parquet \
  --out outputs/eval_v2_stage3_fine || exit 1
echo "CHAIN DONE"
