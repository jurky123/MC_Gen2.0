"""Batch-annotate texture tiles with a local VLM into structured JSON.

Reads a processed manifest (one JSON record per tile, ``src/data/..._processor.py``
format), builds the single + tiled nearest-neighbour views, queries the annotator
and appends one JSON line per tile. Resumable: existing ``source`` entries in the
output are skipped.

Examples:

    # inspect the exact prompt for the first pending tile, no requests
    python scripts/annotate_textures.py --dry-run

    # annotate the James-A fine-tune set against a local vLLM server
    python scripts/annotate_textures.py \
        --manifest data/processed/minecraft_16x_finetune32/manifest.jsonl \
        --out data/build/mc_finetune_annotations/annotations.jsonl

    # in-process (no server), smaller test slice
    python scripts/annotate_textures.py --backend transformers --limit 20

    # bulk coarse labelling of the Stage 2 mmap build
    python scripts/make_annotation_manifest.py --build data/build/stage2_32
    python scripts/annotate_textures.py --profile coarse \
        --manifest data/build/stage2_32/annotation_manifest.jsonl \
        --out data/build/stage2_32/coarse_annotations.jsonl
"""
import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_yaml  # noqa: E402
from data.vlm_caption import AnnotatorConfig, run_annotation  # noqa: E402


def build_config(args) -> AnnotatorConfig:
    cfg = AnnotatorConfig.from_yaml(args.config)
    if args.profile:
        overrides = cfg.profiles.get(args.profile)
        if overrides is None:
            raise SystemExit(f"unknown profile {args.profile!r}; "
                             f"available: {sorted(cfg.profiles)}")
        nested = dict(overrides)
        if isinstance(nested.get("image"), dict):
            nested["image"] = {**cfg.image, **nested["image"]}
        for key, value in nested.items():
            if key in AnnotatorConfig.__dataclass_fields__ and key != "profiles":
                setattr(cfg, key, value)
        cfg.profile = args.profile
    overrides = {
        "backend": args.backend,
        "endpoint": args.endpoint,
        "model": args.model,
        "served_model_name": args.served_model_name,
    }
    for key, value in overrides.items():
        if value:
            setattr(cfg, key, value)
    if args.concurrency:
        cfg.concurrency = args.concurrency
    if args.max_tokens:
        cfg.max_tokens = args.max_tokens
    if args.endpoints:
        cfg.endpoints = list(args.endpoints)
    return cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "annotator.yaml"))
    ap.add_argument("--profile", default="",
                    help="named profile from the config (e.g. coarse, fine)")
    ap.add_argument("--manifest",
                    default=str(ROOT / "data" / "processed" /
                                "minecraft_16x_finetune32" / "manifest.jsonl"))
    ap.add_argument("--out",
                    default=str(ROOT / "data" / "build" /
                                "mc_finetune_annotations" / "annotations.jsonl"))
    ap.add_argument("--log-file", default="",
                    help="progress log path; default is <out>.log (append, tail -f)")
    ap.add_argument("--root", default=str(ROOT),
                    help="base dir for relative image paths in the manifest")
    ap.add_argument("--backend", default="")
    ap.add_argument("--endpoint", default="")
    ap.add_argument("--endpoints", nargs="+", default=[],
                    help="multiple OpenAI-compatible endpoints; shared-queue "
                         "data parallel (one per GPU, keeps both busy)")
    ap.add_argument("--model", default="")
    ap.add_argument("--served-model-name", default="")
    ap.add_argument("--concurrency", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--no-resume", action="store_true",
                    help="re-annotate records already present in --out")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="split the pending work across N parallel workers (data parallel)")
    ap.add_argument("--shard-index", type=int, default=0,
                    help="this worker's shard in [0, num-shards)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    manifest = Path(args.manifest)
    if not manifest.exists():
        raise SystemExit(f"manifest not found: {manifest}")

    cfg = build_config(args)
    out_path = Path(args.out)
    if args.num_shards > 1:
        out_path = out_path.with_name(
            f"{out_path.stem}.shard{args.shard_index:02d}{out_path.suffix}")
    log_file = Path(args.log_file) if args.log_file else out_path.with_suffix(".log")
    print(f"progress log: {log_file}  (watch: tail -f {log_file})", flush=True)
    summary = asyncio.run(run_annotation(
        cfg,
        manifest=manifest,
        out_path=out_path,
        root=Path(args.root),
        limit=args.limit,
        start=args.start,
        resume=not args.no_resume,
        dry_run=args.dry_run,
        device=args.device,
        dtype=args.dtype,
        log_file=log_file,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    ))
    print(summary)


if __name__ == "__main__":
    main()
