#!/bin/bash
# Queue v3: finish the missing MC->HD rows -> rebuild pairs (ref 128) ->
# train stylizer v3 -> Gate-1 eval.
set -x
cd /home/iflab/models/MC_Gen2.0
PY=/home/iflab/miniconda3/envs/mcgen/bin/python

while pgrep -f "[b]uild_mchd_pairs.py" > /dev/null; do sleep 60; done
sleep 5
echo "counts:"; ls pairs/mchd_stage3_a/hd | wc -l; ls pairs/mchd_stage3_b/hd | wc -l; \
               ls pairs/mchd_fill_a/hd | wc -l; ls pairs/mchd_fill_b/hd | wc -l

$PY scripts/build_stylizer_pairs_v2.py \
  --sources pairs/mchd_stage3_a,pairs/mchd_stage3_b,pairs/mchd_fill_a,pairs/mchd_fill_b \
  --ref-size 128 --out pairs/stylizer_v3 || exit 1

$PY scripts/train.py \
  --model configs/model/stylizer_v3.yaml \
  --train configs/train/stylizer_v3.yaml \
  --init-from checkpoints/stage_3_frozen_v2/best.pt || exit 1

$PY scripts/eval_stylizer.py \
  --ckpt checkpoints/stylizer_v3/best.pt --pairs pairs/stylizer_v3 \
  --out outputs/eval_stylizer_v3 || exit 1

$PY scripts/review_stylizer.py \
  --ckpt checkpoints/stylizer_v3/best.pt --pairs pairs/stylizer_v3 \
  --out /tmp/opencode/styl_v3_review || exit 1
echo QUEUE-V3-DONE
