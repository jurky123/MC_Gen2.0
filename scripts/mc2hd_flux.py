"""Tier-A anchor prototype: real MC (32px) -> nearest upscale -> FLUX.2-klein-4B
image editing -> HD reference. The MC target stays the supervision; the HD may
carry reasonable hallucination but must pass structure filtering.

    python scripts/mc2hd_flux.py --rows 156911,474710 --out pairs/anch_proto
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FLUX_MODEL = "/home/iflab/models/FLUX.2-klein-4B"
FLUX_LICENSE = "Apache-2.0 (black-forest-labs/FLUX.2-klein-4B)"

EDIT_PROMPT = ("Redraw this pixel-art game texture as a smooth, highly detailed "
               "high-resolution game texture with realistic materials. Keep "
               "the exact same layout, shapes, colors, materials and "
               "structure, but REMOVE all pixelation and blockiness: smooth "
               "gradients, clean anti-aliased edges, fine surface detail. "
               "NOT voxel, NOT minecraft style, NOT made of cubes. "
               "No background scene, no text, no watermark.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", default="pairs/anch_proto")
    ap.add_argument("--build", default=str(ROOT / "data/build/mc_text2image32_wl"))
    ap.add_argument("--prompt-parquet", default="")
    ap.add_argument("--hd-size", type=int, default=512)
    ap.add_argument("--ref-size", type=int, default=64)
    ap.add_argument("--flux-steps", type=int, default=8)
    ap.add_argument("--init-mode", default="nearest",
                    help="upscale filter for the MC init: nearest (edges) or bicubic (smooth)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--flux-device", default="cuda:1")
    args = ap.parse_args()

    import pandas as pd

    out = Path(args.out)
    (out / "hd").mkdir(parents=True, exist_ok=True)
    (out / "ref64").mkdir(parents=True, exist_ok=True)
    build = Path(args.build)
    rows = [int(x) for x in args.rows.split(",") if x.strip()]
    meta = pd.read_parquet(build / "metadata.parquet")
    n = len(meta)
    imgs = np.memmap(build / "images.uint8.mmap", dtype=np.uint8, mode="r",
                     shape=(n, 32, 32, 4))
    pp = args.prompt_parquet or str(build / "stage3_prompts.parquet")
    gp = pd.read_parquet(pp, columns=["prompt_0"])
    prompts = [str(gp["prompt_0"].iloc[i]) for i in rows]

    from diffusers import Flux2KleinPipeline

    pipe = Flux2KleinPipeline.from_pretrained(FLUX_MODEL, torch_dtype=torch.bfloat16)
    pipe.to(args.flux_device)
    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass

    man = (out / "manifest.jsonl").open("w", encoding="utf-8")
    with torch.no_grad():
        for j, (row, prompt) in enumerate(zip(rows, prompts)):
            mc = Image.fromarray(np.asarray(imgs[row]), "RGBA")
            bg = Image.new("RGB", mc.size, (255, 255, 255))
            bg.paste(mc.convert("RGB"), mask=mc.split()[3])
            filt = Image.Resampling.BICUBIC if args.init_mode == "bicubic" else Image.Resampling.NEAREST
            init = bg.resize((args.hd_size, args.hd_size), filt)
            g = torch.Generator(device=args.flux_device).manual_seed(args.seed + j)
            t0 = time.time()
            hd = pipe(image=init, prompt=EDIT_PROMPT, height=args.hd_size,
                      width=args.hd_size, num_inference_steps=args.flux_steps,
                      generator=g).images[0]
            el = time.time() - t0
            hd.save(out / "hd" / f"row{row}.png")
            hd.convert("RGB").resize((args.ref_size, args.ref_size),
                                     Image.Resampling.LANCZOS).save(out / "ref64" / f"row{row}.png")
            man.write(json.dumps({
                "row": row, "prompt": prompt,
                "mc_source": "real MC target (supervision)",
                "hd_png": f"hd/row{row}.png", "ref_png": f"ref64/row{row}.png",
                "generator_model": "black-forest-labs/FLUX.2-klein-4B",
                "generator_license": FLUX_LICENSE,
                "mode": "image-edit from MC upscale",
                "init_mode": args.init_mode,
                "edit_prompt": EDIT_PROMPT,
                "flux_steps": args.flux_steps, "hd_size": args.hd_size,
                "seed": args.seed + j, "hd_seconds": round(el, 1),
            }) + "\n")
            man.flush()
            print(f"[{j}] row{row} :: {prompt[:60]}... hd {el:.1f}s", flush=True)
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
