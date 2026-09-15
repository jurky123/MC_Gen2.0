"""Build MC-Texture-CaptionBench and export the human-review spreadsheet.

Step 1 (sampling):

    python scripts/build_caption_bench.py --size 500

Writes ``data/build/caption_bench/bench.jsonl`` (stratified across texture
categories) and ``bench_summary.json``.

Step 2 (after annotating the bench with the local VLM):

    python scripts/annotate_textures.py \
        --manifest data/build/caption_bench/bench.jsonl \
        --out data/build/caption_bench/annotations.jsonl
    python scripts/build_caption_bench.py --export-review

Then open ``review.csv``, fix any wrong ``gold_*`` cells and set
``review_status`` to ``ok``/``edited``, and run ``scripts/score_caption_bench.py``.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eval.caption_bench import (  # noqa: E402
    category_histogram,
    infer_categories,
    load_jsonl,
    primary_category,
    stratify,
    write_review_csv,
)


def load_source_manifest(path):
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build(manifest_paths, out_dir, size, seed, only_blocks):
    records = []
    seen = set()
    for manifest_path in manifest_paths:
        for record in load_source_manifest(manifest_path):
            source = record.get("source") or record.get("path")
            if source in seen:
                continue
            seen.add(source)
            metadata = record.get("metadata") or {}
            categories = infer_categories(metadata.get("type"))
            record["category"] = categories[0]
            record["categories"] = categories
            records.append(record)

    selected = stratify(records, size=size, seed=seed, only_blocks=only_blocks)
    for index, record in enumerate(selected):
        record["id"] = f"cb{index:05d}"

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bench_path = out_dir / "bench.jsonl"
    with bench_path.open("w", encoding="utf-8") as handle:
        for record in selected:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "available": len(records),
        "selected": len(selected),
        "seed": seed,
        "only_blocks": only_blocks,
        "categories": category_histogram(selected),
        "source_manifests": [str(p) for p in manifest_paths],
    }
    (out_dir / "bench_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return bench_path


def export_review(bench_path, annotations_path, review_path):
    bench = load_jsonl(bench_path)
    annotations_by_source = {}
    for record in load_jsonl(annotations_path):
        source = record.get("source")
        if source:
            annotations_by_source[source] = record
    path = write_review_csv(bench, annotations_by_source, review_path)
    matched = sum(1 for record in bench if record.get("source") in annotations_by_source)
    print(f"review rows={len(bench)} annotations_matched={matched} -> {path}")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", action="append", default=[],
                    help="source JSONL manifest(s); default James-A fine-tune set")
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "build" / "caption_bench"))
    ap.add_argument("--size", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only-blocks", action="store_true",
                    help="sample only 'Block - *' types (skip items)")
    ap.add_argument("--export-review", action="store_true",
                    help="export review.csv from bench.jsonl + annotations.jsonl")
    ap.add_argument("--bench", default="")
    ap.add_argument("--annotations", default="")
    ap.add_argument("--review-out", default="")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if args.export_review:
        bench_path = Path(args.bench) if args.bench else out_dir / "bench.jsonl"
        annotations = Path(args.annotations) if args.annotations else out_dir / "annotations.jsonl"
        review = Path(args.review_out) if args.review_out else out_dir / "review.csv"
        if not bench_path.exists():
            raise SystemExit(f"bench not found: {bench_path} (run sampling first)")
        if not annotations.exists():
            raise SystemExit(f"annotations not found: {annotations} "
                             "(run scripts/annotate_textures.py first)")
        export_review(bench_path, annotations, review)
        return

    manifests = [Path(p) for p in args.manifest] or [
        ROOT / "data" / "processed" / "minecraft_16x_finetune32" / "manifest.jsonl"
    ]
    for manifest in manifests:
        if not manifest.exists():
            raise SystemExit(f"manifest not found: {manifest}")
    build(manifests, out_dir, size=args.size, seed=args.seed, only_blocks=args.only_blocks)


if __name__ == "__main__":
    main()
