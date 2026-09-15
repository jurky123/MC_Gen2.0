"""Rewrite texture annotations into K diverse prompts with a text LLM.

Reads the coarse/fine annotations (structured fields + captions) and the cleaned
source label, then asks a text LLM for 4 prompts at different abstraction levels
(factual / evocative / thematic / descriptive) with domain boilerplate removed.

    # inspect the prompt for one sample
    python scripts/rewrite_prompts.py --dry-run --limit 1

    # run (resumable, shardable) against a local OpenAI-compatible server
    python scripts/rewrite_prompts.py \
        --annotations data/build/mc_text2image32_wl/coarse_annotations.shard00.jsonl \
                      data/build/mc_text2image32_wl/coarse_annotations.shard01.jsonl \
        --manifest data/build/mc_text2image32_wl/coarse_manifest.jsonl \
        --out data/build/mc_text2image32_wl/prompt_views.jsonl
"""
import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.prompt_diversify import RewriterConfig, run_rewrite  # noqa: E402


def build_config(args) -> RewriterConfig:
    cfg = RewriterConfig.from_yaml(args.config)
    for key, value in {"endpoint": args.endpoint, "model": args.model,
                       "served_model_name": args.served_model_name}.items():
        if value:
            setattr(cfg, key, value)
    if args.concurrency:
        cfg.concurrency = args.concurrency
    if args.temperature is not None:
        cfg.temperature = args.temperature
    if args.endpoints:
        cfg.endpoints = list(args.endpoints)
    return cfg


def main():
    default_dir = ROOT / "data" / "build" / "mc_text2image32_wl"
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "prompt_rewrite.yaml"))
    ap.add_argument("--annotations", nargs="+",
                    default=[str(default_dir / "coarse_annotations.shard00.jsonl"),
                             str(default_dir / "coarse_annotations.shard01.jsonl")])
    ap.add_argument("--manifest", default=str(default_dir / "coarse_manifest.jsonl"))
    ap.add_argument("--out", default=str(default_dir / "prompt_views.jsonl"))
    ap.add_argument("--log-file", default="")
    ap.add_argument("--endpoint", default="")
    ap.add_argument("--endpoints", nargs="+", default=[],
                    help="multiple OpenAI-compatible endpoints; shared-queue "
                         "data parallel (one per GPU, keeps both busy)")
    ap.add_argument("--model", default="")
    ap.add_argument("--served-model-name", default="")
    ap.add_argument("--concurrency", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    annotations = [Path(p) for p in args.annotations]
    for path in annotations:
        if not path.exists():
            raise SystemExit(f"annotations not found: {path}")
    if not Path(args.manifest).exists():
        raise SystemExit(f"manifest not found: {args.manifest}")

    cfg = build_config(args)
    out_path = Path(args.out)
    if args.num_shards > 1:
        out_path = out_path.with_name(
            f"{out_path.stem}.shard{args.shard_index:02d}{out_path.suffix}")
    log_file = Path(args.log_file) if args.log_file else out_path.with_suffix(".log")
    print(f"progress log: {log_file}  (watch: tail -f {log_file})", flush=True)
    summary = asyncio.run(run_rewrite(
        cfg,
        annotation_paths=annotations,
        manifest=Path(args.manifest),
        out_path=out_path,
        limit=args.limit,
        resume=not args.no_resume,
        dry_run=args.dry_run,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        log_file=log_file,
    ))
    print(summary)


if __name__ == "__main__":
    main()
