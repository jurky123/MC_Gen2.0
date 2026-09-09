import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def seam_score(arr, border_width=2):
    arr = np.asarray(arr, dtype=np.float32)
    l = arr[:, :border_width, :]
    r = arr[:, -border_width:, :]
    t = arr[:border_width, :, :]
    b = arr[-border_width:, :, :]
    h = np.abs(l - r).mean()
    v = np.abs(t - b).mean()
    return 0.5 * (h + v), h, v


def make_tiled_preview(arr, repeats=4):
    arr = np.asarray(arr)
    tiles = np.tile(arr, (repeats, repeats, 1)) if arr.ndim == 3 else np.tile(arr, (repeats, repeats))
    return tiles


def report_directory(png_dir, out_json=None):
    png_dir = Path(png_dir)
    rows = []
    for f in sorted(png_dir.glob("*.png")):
        if ".tiled." in f.name:
            continue
        arr = np.asarray(Image.open(f).convert("RGB"))
        score, h, v = seam_score(arr)
        rows.append({"file": str(f), "seam": float(score), "h": float(h), "v": float(v)})
        Image.fromarray(make_tiled_preview(arr)).save(f.with_suffix(".tiled.png"))
    report = {"samples": len(rows), "mean_seam": float(np.mean([r["seam"] for r in rows])) if rows else None, "rows": rows}
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