"""QC sheet for the merged multi-generator Stylizer pairs.

For N sampled rows (half opaque, half cutout) shows:

    MC target | FLUX HD ref | Qwen HD ref | prompt

so a human can verify both generators map to the same MC target and that
cutouts keep their silhouette in both.

    python scripts/qc_stylizer_pairs_v4.py --n 8 --out /tmp/opencode/qc_v4
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

BUILD = ROOT / "data/build/mc_text2image32_wl"


def on_white(im):
    im = im.convert("RGBA")
    bg = Image.new("RGB", im.size, (255, 255, 255))
    bg.paste(im.convert("RGB"), mask=im.split()[3])
    return bg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=str(ROOT / "pairs/stylizer_v4"))
    ap.add_argument("--n", type=int, default=8, help="rows per class (opaque/cutout)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/tmp/opencode/qc_v4")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pdir = Path(args.pairs)
    df = pd.read_parquet(pdir / "metadata.parquet")
    if "generator" not in df.columns:
        raise SystemExit("metadata has no 'generator' column; not a v4 pair set")

    meta = pd.read_parquet(BUILD / "metadata.parquet")
    imgs = np.memmap(BUILD / "images.uint8.mmap", dtype=np.uint8, mode="r",
                     shape=(len(meta), 32, 32, 4))

    # pick rows that have both generators, half opaque / half cutout
    by_row = {}
    for i, r in df.iterrows():
        by_row.setdefault(int(r["row"]), {})[str(r["generator"])] = i
    both = {row: g for row, g in by_row.items() if len(g) >= 2}
    a0 = {row: float(np.mean(np.asarray(imgs[row])[..., 3] == 0)) for row in both}
    opaque = sorted([r for r in both if a0[r] <= 0.05])
    cutout = sorted([r for r in both if a0[r] > 0.05])
    rng = np.random.RandomState(args.seed)
    rows = (list(rng.choice(opaque, size=min(args.n, len(opaque)), replace=False))
            + list(rng.choice(cutout, size=min(args.n, len(cutout)), replace=False)))
    rows = [int(r) for r in rows]
    if not rows:
        raise SystemExit("no rows with both generators")

    S, gap, hdr = 192, 10, 40
    cols = ["MC target (32)", "FLUX HD ref", "Qwen HD ref"]
    sheet = Image.new("RGB", (3 * S + 4 * gap, len(rows) * (S + hdr + 6) + gap + hdr),
                      (25, 25, 25))
    d = ImageDraw.Draw(sheet)
    d.text((gap, 4), "merged stylizer_v4 pairs:  MC | FLUX.2-klein | Qwen-Image-2.1",
           fill=(255, 255, 0))
    d.text((gap + 3 * S + 4 * gap - 140, 4), f"n_rows={len(rows)}", fill=(180, 180, 180))
    for c, name in enumerate(cols):
        d.text((gap + c * (S + gap), hdr - 2), name, fill=(150, 220, 255))

    for r_i, row in enumerate(rows):
        y = hdr + 6 + r_i * (S + hdr + 6)
        gidx = by_row[row]
        prompt = str(df["prompt"].iloc[gidx[next(iter(gidx))]])
        d.text((gap, y), f"row {row}  a0={a0[row]:.2f}  {prompt[:88]}", fill=(200, 200, 200))
        mc = Image.fromarray(np.asarray(imgs[row]), "RGBA").resize(
            (S, S), Image.Resampling.NEAREST)
        cells = [on_white(mc)]
        for gen in ("flux", "qwen"):
            if gen in gidx:
                hd_path = None
                src = str(df["src"].iloc[gidx[gen]])
                cand = ROOT / src / f"hd/row{row:07d}.png"
                if cand.exists():
                    hd_path = cand
                if hd_path is None:
                    cells.append(Image.new("RGB", (S, S), (80, 0, 0)))
                else:
                    im = Image.open(hd_path)
                    if im.mode == "RGBA":
                        im = on_white(im)
                    else:
                        im = im.convert("RGB")
                    cells.append(im.resize((S, S), Image.Resampling.LANCZOS))
            else:
                cells.append(Image.new("RGB", (S, S), (80, 0, 0)))
        for c, im in enumerate(cells):
            sheet.paste(im, (gap + c * (S + gap), y + 14))

    sheet.save(out / "qc_v4.png")
    print(f"saved {out / 'qc_v4.png'}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
