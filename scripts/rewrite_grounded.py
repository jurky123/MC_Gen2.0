"""Grounded label -> prompt rewrite over a built dataset.

Reads source labels from ``metadata.parquet`` (file_name + type), rewrites them
into one strict prompt per row, and writes a JSONL (resumable, shardable) plus
a merged ``grounded_prompts.parquet`` aligned to the dataset row order.

    # dry-run one row
    python scripts/rewrite_grounded.py --dry-run --limit 1

    # run (2 shards, one per endpoint)
    python scripts/rewrite_grounded.py --num-shards 2 --shard-index 0 \
        --endpoints http://127.0.0.1:8000/v1/chat/completions

    # merge shards into the training parquet
    python scripts/rewrite_grounded.py --merge
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from data.filename_prompts import (  # noqa: E402
    MOD_NOISE, fallback_prompt, label_tokens, type_word,
)
from data.grounded_rewrite import GroundedRewriteConfig, run_grounded_rewrite  # noqa: E402


def build_records(build_dir: Path):
    meta = pd.read_parquet(build_dir / "metadata.parquet")
    records = []
    for i, row in enumerate(meta.to_dict(orient="records")):
        tokens = label_tokens(row, MOD_NOISE)
        records.append({
            "index": i,
            "source": f"{build_dir.name}#{i}",
            "file_name": row.get("file_name"),
            "tokens": tokens,
            "type_word": type_word(row),
        })
    return meta, records


def merge(build_dir: Path, jsonl_paths, out_path: Path, label_by_index):
    prompts = {}
    for p in jsonl_paths:
        p = Path(p)
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("index") is not None and r.get("prompt"):
                    prompts[int(r["index"])] = r["prompt"]
    meta = pd.read_parquet(build_dir / "metadata.parquet", columns=["index"])
    indexes = meta["index"].tolist() if "index" in meta.columns else list(range(len(meta)))
    out = []
    missing = 0
    for i in indexes:
        i = int(i)
        p = prompts.get(i)
        if not p:
            p = fallback_prompt(*label_by_index[i])
            missing += 1
        out.append(p)
    df = pd.DataFrame({"index": indexes, "prompt_0": out, "style_0": ["grounded"] * len(indexes)})
    df.to_parquet(out_path, index=False)
    print(f"wrote {out_path} rows={len(df)} missing(used fallback)={missing}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "grounded_rewrite.yaml"))
    ap.add_argument("--build", default=str(ROOT / "data" / "build" / "mc_text2image32_wl"))
    ap.add_argument("--out", default="")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--jsonl", nargs="*", default=[])
    ap.add_argument("--model", default="")
    ap.add_argument("--endpoints", nargs="*", default=[])
    ap.add_argument("--concurrency", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--log-file", default="")
    args = ap.parse_args()

    build_dir = Path(args.build)
    out_path = Path(args.out) if args.out else build_dir / "grounded_prompts.jsonl"

    if args.merge:
        jsonls = args.jsonl or [build_dir / "grounded_prompts.shard00.jsonl",
                                build_dir / "grounded_prompts.shard01.jsonl"]
        print("building labels for fallback ...")
        _, records = build_records(build_dir)
        label_by_index = {r["index"]: (r["tokens"], r["type_word"]) for r in records}
        merge(build_dir, jsonls, build_dir / "grounded_prompts.parquet", label_by_index)
        return

    cfg = GroundedRewriteConfig.from_yaml(args.config)
    if args.model:
        cfg.model = args.model
    if args.endpoints:
        cfg.endpoints = list(args.endpoints)
    if args.concurrency:
        cfg.concurrency = args.concurrency

    print("building labels ...")
    _, records = build_records(build_dir)
    print(f"records={len(records)}")
    actual_out = out_path
    if args.num_shards > 1:
        actual_out = out_path.with_name(f"{out_path.stem}.shard{args.shard_index:02d}{out_path.suffix}")
    log_file = Path(args.log_file) if args.log_file else actual_out.with_suffix(".log")
    print(f"progress log: {log_file}  (watch: tail -f {log_file})")
    summary = asyncio.run(run_grounded_rewrite(
        cfg, records, actual_out, limit=args.limit, resume=not args.no_resume,
        dry_run=args.dry_run, num_shards=args.num_shards, shard_index=args.shard_index,
        log_file=log_file,
    ))
    print(summary)


if __name__ == "__main__":
    main()
