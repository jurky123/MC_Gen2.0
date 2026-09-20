#!/bin/bash
# Queue: wait for MC->HD production -> rebuild stylizer pairs -> train -> eval.
set -x
cd /home/iflab/models/MC_Gen2.0
PY=/home/iflab/miniconda3/envs/mcgen/bin/python

# 1) wait for both production shards
while pgrep -f "[b]uild_mchd_pairs.py" > /dev/null; do sleep 60; done
sleep 5
ls pairs/mchd_stage3_a/hd | wc -l; ls pairs/mchd_stage3_b/hd | wc -l

# 2) build the pair dataset (ref64 / real MC target)
$PY scripts/build_stylizer_pairs_v2.py --out pairs/stylizer_v2 || exit 1

# 3) Stylizer training (Phase 1, frozen backbone)
$PY scripts/train.py \
  --model configs/model/stylizer.yaml \
  --train configs/train/stylizer_phase1_v2.yaml \
  --init-from checkpoints/stage_3_frozen_v2/best.pt || exit 1

# 4) Gate-1 evaluation
$PY scripts/eval_stylizer.py \
  --ckpt checkpoints/stylizer_phase1_v2/best.pt \
  --pairs pairs/stylizer_v2 \
  --out outputs/eval_stylizer_v2 || exit 1
echo QUEUE-DONE
