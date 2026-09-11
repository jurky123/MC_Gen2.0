import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


def nearest_neighbors(img_arr, mem, k=3):
    arr = np.asarray(img_arr, dtype=np.float32).reshape(-1)
    flat = np.asarray(mem).reshape(mem.shape[0], -1).astype(np.float32)
    d = np.abs(flat - arr[None, :]).mean(axis=1)
    order = np.argsort(d)[:k]
    return [(int(i), float(d[i])) for i in order]


def report_directory(png_dir, mmap_images, image_size=32, out_json=None, channels=3):
    png_dir = Path(png_dir)
    files = [f for f in sorted(png_dir.glob("*.png")) if ".tiled." not in f.name]
    bytes_per = image_size * image_size * channels
    n = os.path.getsize(mmap_images) // bytes_per
    mem = np.memmap(mmap_images, dtype=np.uint8, mode="r", shape=(n, image_size, image_size, channels))
    if channels != 3:
        mem = np.ascontiguousarray(mem[..., :3])
    rows = []
    for f in files:
        arr = np.asarray(Image.open(f).convert("RGB"))
        if arr.shape != (image_size, image_size, 3):
            continue
        rows.append({"file": str(f), "neighbors": nearest_neighbors(arr, mem)})
    report = {"samples": len(rows), "rows": rows}
    if out_json:
        Path(out_json).write_text(json.dumps(report, indent=2))
    else:
        print(json.dumps(report, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("mmap")
    ap.add_argument("--out", default="")
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--channels", type=int, default=3, choices=(3, 4))
    args = ap.parse_args()
    report_directory(args.dir, args.mmap, image_size=args.size, out_json=args.out or None, channels=args.channels)


if __name__ == "__main__":
    main()