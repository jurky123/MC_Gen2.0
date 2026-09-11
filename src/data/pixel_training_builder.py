"""Build a training mmap from processed pixel manifests."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def split_for(container):
    value = int(hashlib.sha1(container.encode()).hexdigest()[:8], 16) % 100
    if value < 90:
        return "train"
    if value < 95:
        return "val"
    return "test"


def load_records(manifests):
    records = []
    seen = set()
    for manifest in manifests:
        with open(manifest, encoding="utf-8") as handle:
            for line in handle:
                rec = json.loads(line)
                digest = Path(rec["path"]).stem
                if digest in seen:
                    continue
                seen.add(digest)
                rec["source_set"] = Path(manifest).parent.name
                records.append(rec)
    return records


def build(manifests, out, size=32):
    records = load_records(manifests)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    images = np.memmap(
        out / "images.uint8.mmap", dtype=np.uint8, mode="w+",
        shape=(len(records), size, size, 3),
    )
    splits = {"train": [], "val": [], "test": []}
    metadata = []
    for index, rec in enumerate(records):
        image = Image.open(rec["path"]).convert("RGBA")
        background = Image.new("RGBA", image.size, (0, 0, 0, 255))
        background.alpha_composite(image)
        rgb = background.convert("RGB").resize((size, size), Image.Resampling.NEAREST)
        images[index] = np.asarray(rgb, dtype=np.uint8)
        group = str(rec.get("container", rec.get("source", "")))
        splits[split_for(group)].append(index)
        rec["index"] = index
        rec["weak_prompt"] = "pixel art game asset"
        metadata.append({k: v for k, v in rec.items() if k != "path"})
        if (index + 1) % 10000 == 0:
            print(f"wrote {index + 1}/{len(records)}", flush=True)
    images.flush()
    pd.DataFrame(metadata).to_parquet(out / "metadata.parquet", index=False)
    with open(out / "splits.json", "w") as handle:
        json.dump(splits, handle)
    print(f"done rows={len(records)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifests", nargs="+")
    parser.add_argument("--out", default="data/build/pixel_stage1_32")
    parser.add_argument("--size", type=int, default=32)
    args = parser.parse_args()
    build(args.manifests, args.out, args.size)
