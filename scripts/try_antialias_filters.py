"""Anti-aliasing / low-pass experiments for MC textures (visual only).

Each MC texture (32px, native 16x16 grid) is upscaled to 512 with NEAREST
(16px hard blocks) and then filtered by various operations that suppress the
stair-step edges while keeping low-frequency structure and colour.

    python scripts/try_antialias_filters.py --rows r1,r2,... --out /tmp/opencode/aa
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def up_nearest(arr_rgba, size):
    im = Image.fromarray(arr_rgba, "RGBA")
    return np.asarray(im.resize((size, size), Image.Resampling.NEAREST))


def gauss(rgba, sigma):
    rgb = Image.fromarray(rgba[..., :3])
    a = Image.fromarray(rgba[..., 3])
    rgb = rgb.filter(ImageFilter.GaussianBlur(sigma))
    a = a.filter(ImageFilter.GaussianBlur(sigma * 0.5))
    return np.dstack([np.asarray(rgb), np.asarray(a)])


def lanczos(rgba, size, small=32):
    im = Image.fromarray(rgba, "RGBA").resize((small, small), Image.Resampling.LANCZOS)
    return np.asarray(im.resize((size, size), Image.Resampling.LANCZOS))


def downup(rgba, size, mid=128):
    im = Image.fromarray(rgba, "RGBA").resize((mid, mid), Image.Resampling.LANCZOS)
    return np.asarray(im.resize((size, size), Image.Resampling.BICUBIC))


def bilateral(rgba, d=25, sc=60, ss=60):
    rgb = cv2.bilateralFilter(rgba[..., :3], d, sc, ss)
    a = cv2.bilateralFilter(rgba[..., 3], d, sc, ss)
    return np.dstack([rgb, a])


def meanshift(rgba, sp=20, sr=40):
    rgb = cv2.pyrMeanShiftFiltering(rgba[..., :3], sp, sr)
    a = cv2.pyrMeanShiftFiltering(cv2.cvtColor(rgba[..., 3], cv2.COLOR_GRAY2BGR), sp, sr)[..., 0]
    return np.dstack([rgb, a])


def median(rgba, k=21):
    rgb = cv2.medianBlur(rgba[..., :3], k)
    a = cv2.medianBlur(rgba[..., 3], k)
    return np.dstack([rgb, a])


def guided(rgba, radius=16, eps=1e-2):
    guide = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    out = cv2.ximgproc.guidedFilter(guide, rgba[..., :3].astype(np.float32) / 255.0,
                                    radius, eps) * 255.0
    a = cv2.ximgproc.guidedFilter(guide, rgba[..., 3].astype(np.float32) / 255.0,
                                  radius, eps) * 255.0
    return np.dstack([out.astype(np.uint8), a.astype(np.uint8)])


def bil_gauss(rgba, sigma=12):
    """Bilateral then mild gaussian: smooth stair corners, keep colour edges."""
    b = bilateral(rgba, d=25, sc=70, ss=70)
    return gauss(b, sigma)


VARIANTS = {
    "nearest": lambda r, s: r,
    "gauss8": lambda r, s: gauss(r, 8),
    "gauss16": lambda r, s: gauss(r, 16),
    "gauss28": lambda r, s: gauss(r, 28),
    "lanczos": lambda r, s: lanczos(r, s),
    "down128up": lambda r, s: downup(r, s, 128),
    "down64up": lambda r, s: downup(r, s, 64),
    "bilateral": lambda r, s: bilateral(r),
    "meanshift": lambda r, s: meanshift(r),
    "bil+gauss12": lambda r, s: bil_gauss(r, 12),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--crop", type=int, default=160, help="zoom crop size for the edge sheet")
    ap.add_argument("--build", default=str(ROOT / "data/build/mc_text2image32_wl"))
    ap.add_argument("--out", default="/tmp/opencode/aa")
    args = ap.parse_args()

    import pandas as pd

    build = Path(args.build)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_parquet(build / "metadata.parquet")
    imgs = np.memmap(build / "images.uint8.mmap", dtype=np.uint8, mode="r",
                     shape=(len(meta), 32, 32, 4))
    if args.rows:
        rows = [int(x) for x in args.rows.split(",") if x.strip()]
    else:
        s3 = json.loads((build / "stage3_splits.json").read_text())
        cand = s3["val"]
        bl = [i for i in cand if str(meta.at[i, "type"]) == "block"][:args.n // 2]
        it = [i for i in cand if str(meta.at[i, "type"]) == "item"][:args.n - args.n // 2]
        rows = bl + it

    S = 180
    names = list(VARIANTS)
    sheet = Image.new("RGB", (len(names) * S + (len(names) + 1) * 4,
                              len(rows) * (S + 4) + 60), (25, 25, 25))
    d = ImageDraw.Draw(sheet)
    for c, n in enumerate(names):
        d.text((4 + c * (S + 4) + 4, 6), n, fill=(255, 255, 0))
    edge_sheet = Image.new("RGB", (len(names) * S + (len(names) + 1) * 4,
                                   args.crop * 2 + 60), (25, 25, 25))
    de = ImageDraw.Draw(edge_sheet)
    for c, n in enumerate(names):
        de.text((4 + c * (S + 4) + 4, 6), n, fill=(255, 255, 0))

    for r, row in enumerate(rows):
        base = up_nearest(np.asarray(imgs[row]), args.size)
        y = 40 + r * (S + 4)
        for c, (n, fn) in enumerate(VARIANTS.items()):
            try:
                if n == "guided":
                    v = guided(base)
                else:
                    v = fn(base, args.size)
            except Exception as exc:
                print(f"{n} failed: {exc}")
                v = base
            pil = Image.fromarray(v[..., :4].astype(np.uint8), "RGBA")
            bg = Image.new("RGB", pil.size, (255, 255, 255))
            bg.paste(pil.convert("RGB"), mask=pil.split()[3])
            sheet.paste(bg.resize((S, S), Image.Resampling.LANCZOS), (4 + c * (S + 4), y))
        # zoom crop from the centre for edge inspection
        if r == 0:
            for c, (n, fn) in enumerate(VARIANTS.items()):
                try:
                    v = fn(base, args.size) if n != "guided" else guided(base)
                except Exception:
                    v = base
                pil = Image.fromarray(v[..., :4].astype(np.uint8), "RGBA")
                bg = Image.new("RGB", pil.size, (255, 255, 255))
                bg.paste(pil.convert("RGB"), mask=pil.split()[3])
                cx = (args.size - args.crop) // 2
                crop = bg.crop((cx, cx, cx + args.crop, cx + args.crop))
                edge_sheet.paste(crop.resize((S, S), Image.Resampling.NEAREST),
                                 (4 + c * (S + 4), 40))
    sheet.save(out / "aa_sheet.png")
    edge_sheet.save(out / "aa_edges.png")
    print("saved", out / "aa_sheet.png", "and", out / "aa_edges.png")
    print("rows:", rows)


if __name__ == "__main__":
    main()
