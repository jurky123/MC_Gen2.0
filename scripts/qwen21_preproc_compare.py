"""Qwen-Image-2.1 MC->HD: compare input preprocessing variants (+ negative prompt).

Variants (same as the FLUX study): nearest, gauss16, gauss28, bil+gauss12.
A FLUX.2-klein column (bil+gauss12) is included for reference.
Qwen-Image-2.1 is research-licensed (non-commercial).

    python scripts/qwen21_preproc_compare.py --n 6 --size 384
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_mchd_pairs import REALISM, clean_subject  # noqa: E402
from try_antialias_filters import bilateral, gauss, up_nearest  # noqa: E402

FLUX_MODEL = "/home/iflab/models/FLUX.2-klein-4B"
QWEN_MODEL = "/home/iflab/models/Qwen-Image-2.1"

NEG = ("voxel, voxels, blocky, cubes, cubic, minecraft, pixel art, pixelated, "
       "low resolution, jagged stair-step edges, mosaic, grid, aliasing, flat "
       "untextured colors, background scene, text, watermark, logo")


def variants(mc_rgba, size):
    pil = Image.fromarray(np.asarray(mc_rgba, dtype=np.uint8), "RGBA")
    bg = Image.new("RGB", pil.size, (255, 255, 255))
    bg.paste(pil.convert("RGB"), mask=pil.split()[3])
    base = up_nearest(np.dstack([np.asarray(bg), np.full((32, 32), 255, np.uint8)]), size)
    sc = size / 512.0
    d = max(3, int(round(25 * sc)) | 1)
    bil = bilateral(base, d=d, sc=60, ss=60)
    g = lambda src, s: gauss(src, max(1.0, s * sc))[..., :3].astype(np.uint8)  # noqa: E731
    return {
        "nearest": base[..., :3].astype(np.uint8),
        "gauss16": g(base, 16),
        "gauss28": g(base, 28),
        "bil+g12": g(bil, 12),
        "bil+g16": g(bil, 16),
        "bil+g28": g(bil, 28),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--true-cfg", type=float, default=4.0)
    ap.add_argument("--flux-steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variants", default="nearest,gauss16,gauss28,bil+g12,bil+g16,bil+g28")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--with-flux", action="store_true")
    ap.add_argument("--no-with-flux", dest="with_flux", action="store_false")
    ap.set_defaults(with_flux=False)
    ap.add_argument("--build", default=str(ROOT / "data/build/mc_text2image32_wl"))
    ap.add_argument("--out", default="/tmp/opencode/qwen21_preproc")
    args = ap.parse_args()

    import pandas as pd

    build = Path(args.build)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_parquet(build / "metadata.parquet")
    imgs = np.memmap(build / "images.uint8.mmap", dtype=np.uint8, mode="r",
                     shape=(len(meta), 32, 32, 4))
    s3p = pd.read_parquet(build / "stage3_prompts.parquet", columns=["prompt_0"])
    if args.rows:
        rows = [int(x) for x in args.rows.split(",") if x.strip()]
    else:
        s3 = json.loads((build / "stage3_splits.json").read_text())
        bl = [i for i in s3["val"] if str(meta.at[i, "type"]) == "block"][:args.n // 2]
        it = [i for i in s3["val"] if str(meta.at[i, "type"]) == "item"][:args.n - args.n // 2]
        rows = bl + it

    from diffusers import Flux2KleinPipeline, QwenImage21Pipeline

    qwen = QwenImage21Pipeline.from_pretrained(QWEN_MODEL, torch_dtype=torch.bfloat16)
    qwen.to("cuda:0")
    qwen.set_progress_bar_config(disable=True)
    if args.with_flux:
        flux = Flux2KleinPipeline.from_pretrained(FLUX_MODEL, torch_dtype=torch.bfloat16)
        flux.to("cuda:1")
        flux.set_progress_bar_config(disable=True)

    names = [x.strip() for x in args.variants.split(",") if x.strip()]
    records, tiles = [], []
    with torch.no_grad():
        for row in rows:
            subject = clean_subject(str(s3p["prompt_0"].iloc[row]))
            prompt = f"{subject}. {REALISM}"
            vs = variants(np.asarray(imgs[row]), args.size)
            outs = {}
            for name in names:
                p = out / f"row{row:07d}_qwen21_{name.replace('+','')}.png"
                if args.skip_existing and p.exists():
                    outs[name] = Image.open(p).convert("RGB")
                    continue
                init = Image.fromarray(vs[name])
                g = torch.Generator(device="cuda:0").manual_seed(args.seed + row)
                t0 = time.time()
                hd = qwen(prompt=prompt, image=[init], negative_prompt=NEG,
                          height=args.size, width=args.size,
                          num_inference_steps=args.steps,
                          true_cfg_scale=args.true_cfg, generator=g).images[0]
                s = time.time() - t0
                hd.save(p)
                outs[name] = hd
                print(f"row{row} qwen21/{name}: {s:.1f}s mode={hd.mode}", flush=True)
            if args.with_flux:
                g = torch.Generator(device="cuda:1").manual_seed(args.seed + row)
                hd = flux(image=Image.fromarray(vs["bil+g12"]), prompt=prompt,
                          height=args.size, width=args.size,
                          num_inference_steps=args.flux_steps, generator=g).images[0]
                hd.save(out / f"row{row:07d}_flux_bilg12.png")
                outs["flux"] = hd
            records.append({"row": row, "prompt": prompt, "variants": list(outs),
                            "qwen_negative": NEG, "size": args.size})
            tiles.append((subject, np.asarray(imgs[row]), outs))

    cols = names + (["flux"] if args.with_flux else [])
    S, gap = 230, 6
    sheet = Image.new("RGB", ((len(cols) + 1) * S + (len(cols) + 2) * gap,
                              len(tiles) * (S + 28) + gap + 12), (24, 24, 24))
    d = ImageDraw.Draw(sheet)
    d.text((gap, 2), "raw MC | " + " | ".join(
        (n + " (qwen2.1)") if n != "flux" else "FLUX bil+g12" for n in cols), fill=(255, 255, 0))
    for r, (subj, mc, outs) in enumerate(tiles):
        y = 24 + r * (S + 28)
        d.text((gap, y), subj[:110], fill=(150, 220, 255))
        pil = Image.fromarray(mc, "RGBA")
        bg = Image.new("RGB", pil.size, (255, 255, 255))
        bg.paste(pil.convert("RGB"), mask=pil.split()[3])
        sheet.paste(bg.resize((S, S), Image.Resampling.NEAREST), (gap, y + 24))
        for c, name in enumerate(cols):
            if name not in outs:
                continue
            im = outs[name].convert("RGB").resize((S, S), Image.Resampling.LANCZOS)
            sheet.paste(im, ((c + 1) * S + (c + 2) * gap, y + 24))
    sheet.save(out / "compare.png")
    (out / "manifest.jsonl").write_text("\n".join(json.dumps(x) for x in records))
    print("saved", out / "compare.png")


if __name__ == "__main__":
    main()
