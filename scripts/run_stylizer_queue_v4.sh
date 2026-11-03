#!/bin/bash
# Queue v4: wait for the Qwen cutout rebuild -> merge multi-generator pairs ->
# QC sheet -> train Stylizer v4 (learned null-ref + ref_present) -> Gate-1/OOD eval.
#
#   nohup bash scripts/run_stylizer_queue_v4.sh >/tmp/opencode/queue_v4.log 2>&1 &
#
set -u
cd /home/iflab/models/MC_Gen2.0
PY=/home/iflab/miniconda3/envs/mcgen/bin/python
LOG=/tmp/opencode/queue_v4.log
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

log "waiting for Qwen cutout rebuild (build_mchd_pairs_qwen21)..."
while pgrep -f "[b]uild_mchd_pairs_qwen21" >/dev/null; do sleep 120; done
sleep 5

log "verifying pair counts"
fail=0
check() { # dir expected
  local n; n=$(ls "$1/hd" 2>/dev/null | wc -l)
  if [ "$n" -lt "$2" ]; then log "  FAIL $1: $n < $2"; fail=1; else log "  ok $1: $n"; fi
}
check pairs/mchd_qwen21_cut_a 5533
check pairs/mchd_qwen21_cut_b 5533
check pairs/mchd_cutout_flux_a 5533
check pairs/mchd_cutout_flux_b 5533
if [ "$fail" -ne 0 ]; then log "ABORT: pair dirs incomplete"; exit 1; fi

log "building multi-generator stylizer pairs (ref 128)"
"$PY" scripts/build_stylizer_pairs_v4.py --out pairs/stylizer_v4 --ref-size 128 \
  >>"$LOG" 2>&1 || { log "ABORT: build_stylizer_pairs_v4 failed"; exit 1; }

log "QC sheet"
"$PY" scripts/qc_stylizer_pairs_v4.py --pairs pairs/stylizer_v4 --n 8 \
  --out /tmp/opencode/qc_v4 >>"$LOG" 2>&1 || log "WARN: QC sheet failed"

log "unit tests"
"$PY" -m pytest tests -q >>"$LOG" 2>&1 || { log "ABORT: tests failed"; exit 1; }

log "training Stylizer v4 (4000 steps, frozen backbone, learned null-ref)"
"$PY" scripts/train.py \
  --model configs/model/stylizer_v4.yaml \
  --train configs/train/stylizer_v4.yaml \
  --init-from checkpoints/stage_3_frozen_v2/best.pt >>"$LOG" 2>&1 \
  || { log "ABORT: training failed"; exit 1; }

log "Gate-1 eval"
"$PY" scripts/eval_stylizer.py \
  --ckpt checkpoints/stylizer_v4/best.pt --pairs pairs/stylizer_v4 \
  --out outputs/eval_stylizer_v4 >>"$LOG" 2>&1 || log "WARN: eval_stylizer failed"

log "OOD eval"
"$PY" scripts/eval_ood.py --ckpt checkpoints/stylizer_v4/best.pt \
  --pairs pairs/stylizer_v4 >>"$LOG" 2>&1 || log "WARN: eval_ood failed"

log "reference robustness probe"
"$PY" scripts/probe_reference_robustness.py --ckpt checkpoints/stylizer_v4/best.pt \
  --pairs pairs/stylizer_v4 >>"$LOG" 2>&1 || log "WARN: probe failed"

log "review sheet"
"$PY" scripts/review_stylizer.py --ckpt checkpoints/stylizer_v4/best.pt \
  --pairs pairs/stylizer_v4 --out /tmp/opencode/styl_v4_review >>"$LOG" 2>&1 \
  || log "WARN: review failed"

log "QUEUE-V4-DONE"
