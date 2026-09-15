"""Curate a high-quality annotation manifest from the Stage 2 mmap build.

Stage 2 mixes ~1.03M Modrinth-mod textures: block vs item, many licenses
(including All-Rights-Reserved) and a lot of non-texture assets (particles,
overlays, GUI, fonts, entity sprites). Fine annotation with the 27B model is
expensive, so this selects the subset worth labelling:

    type == block
    license in a permissive/clean set (no ARR / NC / ND / unknown)
    filename not matching non-texture keywords
    image not near-solid (pixel std above a threshold)
    at most N samples per mod project (avoids one mod dominating)
    shuffled, optional global limit

The output JSONL is directly consumable by scripts/annotate_textures.py.

    # inspect the funnel only
    python scripts/curate_annotation_set.py --summary

    # write the curated manifest
    python scripts/curate_annotation_set.py \
        --out data/build/stage2_32/high_quality_manifest.jsonl --max-per-project 300

Then run fine annotation with data parallelism (one worker per GPU):

    CUDA_VISIBLE_DEVICES=0 python scripts/annotate_textures.py --profile fine \
        --num-shards 2 --shard-index 0 \
        --manifest data/build/stage2_32/high_quality_manifest.jsonl \
        --out data/build/stage2_32/fine_annotations.jsonl
    CUDA_VISIBLE_DEVICES=1 python scripts/annotate_textures.py --profile fine \
        --num-shards 2 --shard-index 1 \
        --manifest data/build/stage2_32/high_quality_manifest.jsonl \
        --out data/build/stage2_32/fine_annotations.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from make_annotation_manifest import build_records, infer_shape  # type: ignore  # noqa: E402

CC_GROUPS = {"CC0-1.0", "CC-BY-4.0", "CC-BY-SA-4.0"}
PERMISSIVE = {"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "MPL-2.0", "Zlib", "ISC"}
COPYLEFT = {"GPL-3.0-only", "GPL-3.0-or-later", "GPL-2.0-only", "GPL-2.0-or-later",
            "LGPL-3.0-only", "LGPL-3.0-or-later", "LGPL-2.1-only", "AGPL-3.0-only",
            "EUPL-1.2"}

DEFAULT_EXCLUDE = [
    "gui", "font", "entity", "entities", "particle", "particles",
]


def license_group(license_id, groups):
    license_id = license_id or ""
    if license_id in CC_GROUPS:
        return "cc"
    if license_id in PERMISSIVE:
        return "permissive"
    if license_id in COPYLEFT:
        return "copyleft"
    if license_id.startswith(("CC-BY-NC", "CC-BY-ND")):
        return "nc_nd"
    if license_id.startswith("LicenseRef-All-Rights"):
        return "arr"
    if not license_id:
        return "unknown"
    return "other"


def parse_metadata(df):
    col = next((c for c in ("source_metadata", "metadata") if c in df.columns), None)
    if col is None:
        return df.assign(_meta=[{}] * len(df))
    metas = []
    for raw in df[col].tolist():
        if isinstance(raw, str) and raw.strip().startswith("{"):
            try:
                metas.append(json.loads(raw))
            except json.JSONDecodeError:
                metas.append({})
        else:
            metas.append({})
    return df.assign(_meta=metas)


def image_unique_colors(mmap_path, shape, indices, chunk=8192, dtype=np.uint8):
    """Per-image count of distinct colours (vectorised, mmap-backed)."""
    height, width, channels = shape
    count = mmap_path.stat().st_size // (height * width * channels)
    array = np.memmap(mmap_path, dtype=dtype, mode="r", shape=(count, height, width, channels))
    unique = np.empty(len(indices), dtype=np.int32)
    for start in range(0, len(indices), chunk):
        block = indices[start:start + chunk]
        data = np.asarray(array[block]).reshape(len(block), height * width, channels)
        packed = np.zeros((len(block), height * width), dtype=np.uint32)
        for channel in range(channels):
            packed = (packed << 8) | data[..., channel].astype(np.uint32)
        packed.sort(axis=1)
        unique[start:start + chunk] = 1 + (np.diff(packed, axis=1) != 0).sum(axis=1)
    return unique


def resolve_field(df, meta, name, lower=False):
    """Read a field from a flat column if present, else from parsed JSON meta."""
    if name in df.columns:
        values = df[name].fillna("").astype(str)
    else:
        values = meta.map(lambda d: str(d.get(name, "")))
    return values.str.lower() if lower else values


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", default=str(ROOT / "data" / "build" / "mc_text2image32_wl"))
    ap.add_argument("--out", default="")
    ap.add_argument("--types", default="block,item",
                    help="comma list of types to keep (blank = all)")
    ap.add_argument("--license-groups", default="all",
                    help="cc,permissive,copyleft,nc_nd,arr,unknown,other or 'all'")
    ap.add_argument("--exclude-keywords", default=",".join(DEFAULT_EXCLUDE))
    ap.add_argument("--min-colors", type=int, default=0,
                    help="minimum distinct colours; 0 disables the check")
    ap.add_argument("--max-per-project", type=int, default=0,
                    help="cap samples per mod project; 0 disables the cap")
    ap.add_argument("--shuffle", action="store_true", default=True)
    ap.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--summary", action="store_true", help="print funnel counts only")
    args = ap.parse_args()

    build_dir = Path(args.build)
    df = parse_metadata(pd.read_parquet(build_dir / "metadata.parquet"))
    total = len(df)
    print(f"loaded {total} rows from {build_dir / 'metadata.parquet'}")

    meta = df["_meta"]
    df["_type"] = resolve_field(df, meta, "type", lower=True)
    df["_license"] = resolve_field(df, meta, "license")
    df["_license_group"] = df["_license"].map(lambda l: license_group(l, args.license_groups))
    project = resolve_field(df, meta, "project_id")
    if (project == "").all():
        project = resolve_field(df, meta, "mod_slug")
    df["_project"] = project
    df["_file_name"] = resolve_field(df, meta, "file_name", lower=True)

    keep_types = {t.strip().lower() for t in args.types.split(",") if t.strip()}
    keep_groups = {g.strip() for g in args.license_groups.split(",") if g.strip()}
    license_all = "all" in keep_groups or not keep_groups
    keywords = [k.strip().lower() for k in args.exclude_keywords.split(",") if k.strip()]

    funnel = {}
    funnel["total"] = total
    if keep_types:
        mask = df["_type"].isin(keep_types)
        funnel[f"type in {sorted(keep_types)}"] = int(mask.sum())
    else:
        mask = pd.Series(True, index=df.index)
    if not license_all:
        mask &= df["_license_group"].isin(keep_groups)
        funnel[f"license in {sorted(keep_groups)}"] = int(mask.sum())
    else:
        funnel["license filter"] = int(mask.sum())
    if keywords:
        kw_mask = df["_file_name"].str.contains("|".join(map(str, keywords)), regex=True)
        mask &= ~kw_mask
        funnel[f"not filename {keywords}"] = int(mask.sum())
    else:
        funnel["filename filter"] = int(mask.sum())

    shape = infer_shape(build_dir, total)
    if args.min_colors > 0:
        candidate_idx = df.index[mask]
        colors = image_unique_colors(build_dir / "images.uint8.mmap", shape,
                                     candidate_idx.to_numpy())
        mask.loc[candidate_idx] = colors >= args.min_colors
        funnel[f"unique colors >= {args.min_colors}"] = int(mask.sum())

    kept = df[mask].copy()
    if args.shuffle:
        kept = kept.sample(frac=1.0, random_state=args.seed)
    if args.max_per_project > 0:
        kept = kept.groupby("_project", group_keys=False).head(args.max_per_project)
        funnel[f"<= {args.max_per_project} per project"] = len(kept)
    if args.limit:
        kept = kept.iloc[: args.limit]
        funnel[f"limit {args.limit}"] = len(kept)

    print("\n== funnel ==")
    for key, value in funnel.items():
        print(f"  {key:45s} {value:>9d}")
    print(f"\nprojects kept: {kept['_project'].nunique()}")

    if args.summary or not args.out:
        if not args.out:
            print("\n(no --out given; summary only)")
        return

    positions = [int(p) for p in kept.index]
    mmap_rel = build_dir / "images.uint8.mmap"
    try:
        display_rel = mmap_rel.relative_to(ROOT)
    except ValueError:
        display_rel = mmap_rel
    records = build_records(kept, positions, build_dir, display_rel, shape, build_dir.name)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(records)} high-quality records -> {out_path}")


if __name__ == "__main__":
    main()
