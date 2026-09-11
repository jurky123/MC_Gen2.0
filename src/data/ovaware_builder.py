import argparse
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from data.build_mmap import build_mmap
from data.license_audit import classify

REPO_ID = "OVAWARE/16xModdedMinecraft"
FILES = ["data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet"]


def _download_shards(out_dir, token=None):
    from huggingface_hub import hf_hub_download

    paths = []
    for fn in FILES:
        p = hf_hub_download(
            repo_id=REPO_ID,
            filename=fn,
            repo_type="dataset",
            token=token or os.environ.get("HF_TOKEN"),
            local_dir=str(out_dir / "raw"),
        )
        paths.append(p)
    return paths


def _decode(image):
    if isinstance(image, bytes):
        return Image.open(io.BytesIO(image))
    if isinstance(image, dict):
        b = image.get("bytes")
        if b is not None:
            return Image.open(io.BytesIO(b))
    if isinstance(image, str):
        return Image.open(image)
    raise TypeError(f"unexpected image field type: {type(image)}")


def _row_to_record(image, file_name, type_, license_, project_id, mod_slug, image_size):
    img = _decode(image).convert("RGB")
    if img.size != (16, 16):
        img = img.resize((16, 16), Image.Resampling.NEAREST)
    if image_size != 16:
        img = img.resize((image_size, image_size), Image.Resampling.NEAREST)
    return {
        "file_name": file_name,
        "type": type_,
        "license": license_,
        "project_id": project_id,
        "mod_slug": mod_slug,
    }, img


def run(out_dir, image_size=16, include_review=False, token=None, limit=None, download=True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shards = [Path(p) for p in _download_shards(out_dir, token=token)] if download else []
    if not shards:
        shards = sorted((out_dir / "raw").rglob("*.parquet"))
        shards = [p for p in shards if p.is_file()]
    if not shards:
        raise FileNotFoundError(f"no parquet shards found under {out_dir / 'raw'}")

    staging = out_dir / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    counts = {"allow": 0, "review": 0, "deny": 0}
    records = []
    seen = set()
    for shard in shards:
        table = pq.read_table(shard, columns=["image", "file_name", "type", "license", "project_id", "mod_slug"])
        df = table.to_pandas()
        for _, row in df.iterrows():
            lic = str(row.get("license") or "")
            status = classify(lic)
            if status == "deny" or (status == "review" and not include_review):
                counts["deny" if status == "deny" else "review"] += 1
                continue
            counts[status] += 1
            key = (row.get("project_id"), row.get("file_name"))
            if key in seen:
                continue
            seen.add(key)
            rec, img = _row_to_record(row["image"], row["file_name"], row["type"], lic, row["project_id"], row["mod_slug"], image_size)
            path = staging / f"{len(records):08d}.png"
            img.save(path)
            rec["path"] = str(path)
            records.append(rec)
            if limit and len(records) >= limit:
                break
        if limit and len(records) >= limit:
            break

    print(f"license filter: {counts}")
    if not records:
        print("no records after filtering")
        return
    build_mmap(out_dir, records, image_size=image_size, split_by="project_id")
    print(f"ovaware built: {len(records)} records -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/build/mc_b")
    ap.add_argument("--size", type=int, default=16, help="output resolution (16 native or 32 nearest-upscaled)")
    ap.add_argument("--include-review", action="store_true", help="also include MIT/Apache/GPL (needs manual asset audit)")
    ap.add_argument("--token", default="", help="HF token for the gated repo (or set HF_TOKEN)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-download", action="store_true")
    args = ap.parse_args()
    run(args.out, image_size=args.size, include_review=args.include_review, token=args.token or None, limit=args.limit or None, download=not args.no_download)


if __name__ == "__main__":
    main()