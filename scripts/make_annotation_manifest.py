"""Build an annotation manifest for a built mmap dataset (Stage 1 / Stage 2).

The VLM annotator can read tiles straight from ``images.uint8.mmap`` so we never
materialise millions of PNGs. This script turns a build directory into a JSONL
manifest whose records carry ``mmap`` + ``index`` + ``shape`` and the source
weak label / metadata that the annotator feeds back into the prompt.

    python scripts/make_annotation_manifest.py \
        --build data/build/stage2_32 \
        --out data/build/stage2_32/annotation_manifest.jsonl

Then run coarse annotation (single view, reference label in prompt):

    python scripts/annotate_textures.py \
        --config configs/annotator.yaml --profile coarse \
        --manifest data/build/stage2_32/annotation_manifest.jsonl \
        --out data/build/stage2_32/coarse_annotations.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

META_JSON_COLUMNS = ("source_metadata", "metadata")


def infer_shape(build_dir, rows):
    summary_path = build_dir / "build_summary.json"
    size = channels = None
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        size = summary.get("image_size")
        channels = summary.get("channels")
    if not size or not channels:
        size, channels = 32, 4
    mmap_path = build_dir / "images.uint8.mmap"
    expected = rows * size * size * channels
    actual = mmap_path.stat().st_size if mmap_path.exists() else expected
    if actual != expected:
        for candidate in (1, 3, 4):
            if rows * size * size * candidate == actual:
                channels = candidate
                break
    return [size, size, channels]


def build_records(df, positions, build_dir, mmap_rel, shape, dataset_name):
    metadata_json_col = next((c for c in META_JSON_COLUMNS if c in df.columns), None)
    rows = df.to_dict(orient="records")
    records = []
    for slot, row in enumerate(rows):
        original_index = int(positions[slot])
        metadata = {}
        raw = row.get(metadata_json_col) if metadata_json_col else None
        if isinstance(raw, str) and raw.strip().startswith("{"):
            try:
                metadata.update(json.loads(raw))
            except json.JSONDecodeError:
                pass
        for key, value in row.items():
            if key in META_JSON_COLUMNS or value is None:
                continue
            if isinstance(value, (str, int, float, bool, list, tuple)):
                metadata[key] = value
        weak = str(metadata.get("weak_prompt") or metadata.get("text") or "")
        records.append({
            "id": f"{dataset_name}{original_index:08d}",
            "source": f"{dataset_name}#{original_index}",
            "mmap": str(mmap_rel),
            "index": original_index,
            "shape": list(shape),
            "dataset": dataset_name,
            "weak_prompt": weak,
            "metadata": metadata,
        })
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", default=str(ROOT / "data" / "build" / "stage2_32"))
    ap.add_argument("--out", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shuffle", action="store_true",
                    help="shuffle row order (keep original mmap index for reads)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard-size", type=int, default=0,
                    help="split output into shards of this many records (0 = single file)")
    args = ap.parse_args()

    build_dir = Path(args.build)
    if not build_dir.exists():
        raise SystemExit(f"build dir not found: {build_dir}")
    metadata_path = build_dir / "metadata.parquet"
    if not metadata_path.exists():
        raise SystemExit(f"metadata not found: {metadata_path}")

    df = pd.read_parquet(metadata_path)
    total = len(df)
    if args.shuffle:
        df = df.sample(frac=1.0, random_state=args.seed)
    if args.limit:
        df = df.iloc[: args.limit]
    positions = [int(p) for p in df.index]
    shape = infer_shape(build_dir, total)
    mmap_rel = (build_dir / "images.uint8.mmap")
    try:
        display_rel = mmap_rel.relative_to(ROOT)
    except ValueError:
        display_rel = mmap_rel
    dataset_name = build_dir.name

    records = build_records(df, positions, build_dir, display_rel, shape, dataset_name)
    out_path = Path(args.out) if args.out else build_dir / "annotation_manifest.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.shard_size and args.shard_size > 0:
        written = []
        for shard_index in range(0, len(records), args.shard_size):
            shard = records[shard_index: shard_index + args.shard_size]
            shard_path = out_path.with_name(
                f"{out_path.stem}.{shard_index // args.shard_size:04d}{out_path.suffix}")
            with shard_path.open("w", encoding="utf-8") as handle:
                for record in shard:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written.append(str(shard_path))
        print(f"wrote {len(records)} records in {len(written)} shards under {out_path.parent}")
        for path in written[:3]:
            print("  ", path)
        if len(written) > 3:
            print(f"   ... {len(written) - 3} more")
    else:
        with out_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"wrote {len(records)} records, shape={shape}, mmap={display_rel} -> {out_path}")

    print(f"  sample weak_prompt: {records[0]['weak_prompt']!r}" if records else "  (empty)")


if __name__ == "__main__":
    main()
