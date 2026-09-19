"""Annotate the curated Stage-3 subset with the local 27B VLM.

    # run (resumable, shardable)
    python scripts/annotate_stage3.py \
        --subset data/build/mc_text2image32_wl/stage3_subset.jsonl \
        --out data/build/mc_text2image32_wl/stage3_annotations.jsonl

    # merge annotations into training prompts + subset splits
    python scripts/annotate_stage3.py --merge
"""
import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from data.filename_prompts import MOD_NOISE, label_tokens, type_word  # noqa: E402
from data.stage3_annotate import Stage3Config, run_stage3  # noqa: E402


def build_records(build_dir, subset_path, mmap_rel, shape=(32, 32, 4)):
    meta = pd.read_parquet(build_dir / "metadata.parquet",
                           columns=["file_name", "type", "mod_slug", "project_id"])
    subset = [json.loads(l) for l in open(subset_path, encoding="utf-8") if l.strip()]
    records = []
    for r in subset:
        i = int(r["index"])
        row = meta.iloc[i].to_dict()
        tokens = label_tokens(row, MOD_NOISE)
        records.append({
            "index": i,
            "source": r.get("source", f"{build_dir.name}#{i}"),
            "tokens": tokens,
            "type_word": type_word(row),
            "mmap": str(mmap_rel),
            "shape": list(shape),
        })
    return records


def split_of(i, val_frac=0.1, test_frac=0.1):
    h = int(hashlib.sha1(str(i).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < test_frac:
        return "test"
    if h < test_frac + val_frac:
        return "val"
    return "train"


def merge(build_dir, subset_path, jsonl_paths, out_dir):
    gp = pd.read_parquet(build_dir / "grounded_prompts.parquet", columns=["prompt_0"])
    n = len(gp)
    prompts = gp["prompt_0"].astype(str).tolist()
    subset = [json.loads(l) for l in open(subset_path, encoding="utf-8") if l.strip()]
    subset_idx = [int(r["index"]) for r in subset]

    ann = {}
    for p in jsonl_paths:
        p = Path(p)
        if not p.exists():
            continue
        for line in p.open("r", encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            a = r.get("annotation") or {}
            if r.get("index") is not None and a.get("short_prompt"):
                ann[int(r["index"])] = a["short_prompt"]
    covered = sum(1 for i in subset_idx if i in ann)
    for i in subset_idx:
        if i in ann:
            prompts[i] = ann[i]

    pd.DataFrame({"index": list(range(n)), "prompt_0": prompts,
                  "style_0": ["stage3"] * n}).to_parquet(out_dir / "stage3_prompts.parquet", index=False)
    splits = {"train": [], "val": [], "test": []}
    for i in subset_idx:
        splits[split_of(i)].append(i)
    (out_dir / "stage3_splits.json").write_text(json.dumps(splits), encoding="utf-8")
    print(f"subset={len(subset_idx)} annotated={covered} -> stage3_prompts.parquet; "
          f"splits train/val/test = {len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "stage3_annotator.yaml"))
    ap.add_argument("--build", default=str(ROOT / "data" / "build" / "mc_text2image32_wl"))
    ap.add_argument("--subset", default="")
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
    ap.add_argument("--log-file", default="")
    args = ap.parse_args()

    build = Path(args.build)
    subset = Path(args.subset) if args.subset else build / "stage3_subset.jsonl"
    out = Path(args.out) if args.out else build / "stage3_annotations.jsonl"

    if args.merge:
        jsonls = args.jsonl or [build / "stage3_annotations.shard00.jsonl",
                                build / "stage3_annotations.shard01.jsonl"]
        merge(build, subset, jsonls, build)
        return

    cfg = Stage3Config.from_yaml(args.config)
    if args.model:
        cfg.model = args.model
    if args.endpoints:
        cfg.endpoints = list(args.endpoints)
    if args.concurrency:
        cfg.concurrency = args.concurrency

    records = build_records(build, subset, build / "images.uint8.mmap")
    if args.limit:
        records = records[: args.limit]
    actual = out
    if args.num_shards > 1:
        actual = out.with_name(f"{out.stem}.shard{args.shard_index:02d}{out.suffix}")
    log_file = Path(args.log_file) if args.log_file else actual.with_suffix(".log")
    print(f"records={len(records)} log={log_file}")
    summary = asyncio.run(run_stage3(
        cfg, records, actual, resume=not args.no_resume, num_shards=args.num_shards,
        shard_index=args.shard_index, log_file=log_file))
    print(summary)


if __name__ == "__main__":
    main()
