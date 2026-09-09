import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def rgb_distance_pairwise(images):
    arrs = [np.asarray(Image.open(f).convert("RGB"), dtype=np.float32).reshape(-1) for f in images]
    n = len(arrs)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.abs(arrs[i] - arrs[j]).mean()
            dist[i, j] = dist[j, i] = float(d)
    return dist


def phash_hamming_pairwise(images):
    try:
        import imagehash
    except ImportError:
        return None
    hashes = [imagehash.phash(Image.open(f)) for f in images]
    n = len(hashes)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist[i, j] = dist[j, i] = float(hashes[i] - hashes[j])
    return dist


def report_directory(png_dir, out_json=None):
    png_dir = Path(png_dir)
    files = [f for f in sorted(png_dir.glob("*.png")) if ".tiled." not in f.name]
    rgb = rgb_distance_pairwise(files)
    ph = phash_hamming_pairwise(files)
    report = {
        "samples": len(files),
        "mean_rgb_l2": float(rgb.mean()) if len(files) > 1 else None,
        "mean_phash_hamming": float(ph.mean()) if (len(files) > 1 and ph is not None) else None,
    }
    if out_json:
        Path(out_json).write_text(json.dumps(report, indent=2))
    else:
        print(json.dumps(report, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    report_directory(args.dir, args.out or None)


if __name__ == "__main__":
    main()