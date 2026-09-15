#!/usr/bin/env bash
# Finish the post-annotation pipeline after the rewrite workers exit:
#   1. wait for rewrite workers
#   2. merge shards -> prompt_views.parquet
#   3. stop the vLLM servers to free both GPUs
#   4. preflight the multi-GPU embedding pool (abort early on failure)
#   5. precompute (N, K, 4096) with Qwen3-VL-Embedding-8B across cuda:0,cuda:1
#
#   nohup bash scripts/queue_finish.sh >/dev/null 2>&1 &
#
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PY:-/home/iflab/miniconda3/envs/mcgen/bin/python}"
DIR="data/build/mc_text2image32_wl"
LOG="$DIR/queue_finish.log"
EMBED="${EMBED:-/home/iflab/models/Qwen3-VL-Embedding-8B}"
K="${K:-4}"
DEVICES="${DEVICES:-cuda:0,cuda:1}"

log() { echo "$(date '+%F %T') $*" >> "$LOG"; }

log "waiting for rewrite workers to exit..."
while pgrep -f "[r]ewrite_prompts" >/dev/null; do sleep 30; done
log "rewrite finished; merging shards"
"$PY" scripts/merge_prompt_views.py \
  --views "$DIR/prompt_views.shard00.jsonl" "$DIR/prompt_views.shard01.jsonl" \
  --build-dir "$DIR" --out "$DIR/prompt_views.parquet" >> "$LOG" 2>&1

log "stopping vLLM servers to free GPUs"
for P in 8000 8001; do
  for pid in $(ss -ltnp 2>/dev/null | grep ":$P" | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u); do
    kill "$pid" 2>/dev/null && log "  stopped :$P pid $pid"
  done
done
sleep 25
log "GPU state: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')"

log "preflight multi-GPU pool on $DEVICES"
if ! "$PY" scripts/preflight_embed.py "$EMBED" "$DEVICES" 4096 >> "$LOG" 2>&1; then
  log "preflight FAILED; aborting before full precompute"
  exit 1
fi

log "encoding K=$K views with $EMBED on $DEVICES"
"$PY" scripts/precompute_text.py \
  --config configs/data/stage_2.yaml \
  --views-parquet "$DIR/prompt_views.parquet" --views "$K" \
  --encoder "$EMBED" --encoder-type qwen3vl --devices "$DEVICES" \
  --out "$DIR" >> "$LOG" 2>&1

log "pipeline done"
