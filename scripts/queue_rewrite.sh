#!/usr/bin/env bash
# Full post-annotation pipeline, run in the background:
#   1. wait for the coarse annotation workers to exit
#   2. launch the K-view prompt rewrite on both GPU replicas (data parallel)
#   3. wait for the rewrite workers, then merge the shards into a parquet
#   4. stop the vLLM servers to free the GPUs
#   5. precompute (N, K, 4096) text embeddings with Qwen3-VL-Embedding-8B
# Every step is resumable; re-running skips completed work.
#
#   nohup bash scripts/queue_rewrite.sh >/dev/null 2>&1 &
#
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PY:-/home/iflab/miniconda3/envs/mcgen/bin/python}"
DIR="data/build/mc_text2image32_wl"
OUT="$DIR/prompt_views.jsonl"
LOG="$DIR/queue_rewrite.log"
MODEL="${MODEL:-qwen3-vl-8b}"
EMBED="${EMBED:-/home/iflab/models/Qwen3-VL-Embedding-8B}"
K="${K:-4}"

mkdir -p "$DIR"
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }

log "waiting for annotation workers to exit..."
while pgrep -f "[a]nnotate_textures" >/dev/null; do sleep 60; done
log "annotation finished; launching rewrite (2 shards)"
for S in 0 1; do
  EP=$((8000 + S))
  setsid "$PY" scripts/rewrite_prompts.py \
    --model "$MODEL" \
    --endpoint "http://127.0.0.1:$EP/v1/chat/completions" \
    --num-shards 2 --shard-index "$S" \
    --out "$OUT" >> "$LOG" 2>&1 &
  log "  rewrite shard $S -> :$EP pid $!"
done

log "waiting for rewrite workers to exit..."
while pgrep -f "[r]ewrite_prompts" >/dev/null; do sleep 60; done
log "rewrite finished; merging prompt views"
"$PY" scripts/merge_prompt_views.py \
  --views "$DIR/prompt_views.shard00.jsonl" "$DIR/prompt_views.shard01.jsonl" \
  --build-dir "$DIR" --out "$DIR/prompt_views.parquet" >> "$LOG" 2>&1

log "stopping vLLM servers to free GPUs"
for P in 8000 8001; do
  for pid in $(ss -ltnp 2>/dev/null | grep ":$P" | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u); do
    kill "$pid" 2>/dev/null && log "  stopped :$P pid $pid"
  done
done
sleep 20

log "encoding K=$K views with $EMBED (dynamic multi-GPU pool)"
"$PY" scripts/precompute_text.py \
  --config configs/data/stage_2.yaml \
  --views-parquet "$DIR/prompt_views.parquet" --views "$K" \
  --encoder "$EMBED" --encoder-type qwen3vl --devices "${DEVICES:-cuda:0,cuda:1}" \
  --out "$DIR" >> "$LOG" 2>&1

log "pipeline done"
