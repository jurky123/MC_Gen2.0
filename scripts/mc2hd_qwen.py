"""MC -> HD via Qwen-Image-Edit (image edit), for Path-A style pairs.

Unlike the step-distilled FLUX.2-klein, Qwen-Image-Edit is a CFG model with a
real negative prompt, so voxel/pixelation can be actively suppressed.

    python scripts/mc2hd_qwen.py --rows 156911,474710 --out pairs/anch_qwen
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MODEL = "/home/iflab/models/Qwen-Image-Edit-2511"
LICENSE = "Apache-2.0 (Qwen/Qwen-Image-Edit-2511)"

EDIT_PROMPT = ("Turn this pixel-art game sprite into a smooth, highly detailed "
               "high-resolution game texture. Keep the same object, the same "
               "orientation, the same proportions, the same silhouette and "
               "the same colors. Round off the pixel stair-steps into smooth "
               "continuous shapes. Add realistic surface detail and clean "
               "anti-aliased edges.")
NEGATIVE = ("voxel, voxels, blocky, cube, cubic, minecraft, pixel art, "
            "pixelated, low resolution, jagged stair-step edges, mosaic, "
            "grid, aliasing, flat colors, background scene, text, watermark")


def load_pipe(device):
    from diffusers import QwenImageEditPlusPipeline

    pipe = QwenImageEditPlusPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    pipe.to(device)
    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass
    return pipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", default="pairs/anch_qwen")
    ap.add_argument("--build", default=str(ROOT / "data/build/mc_text2image32_wl"))
    ap.add_argument("--prompt-parquet", default="")
    ap.add_argument("--edit-prompt", default=EDIT_PROMPT)
    ap.add_argument("--negative", default=NEGATIVE)
    ap.add_argument("--hd-size", type=int, default=512)
    ap.add_argument("--ref-size", type=int, default=64)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--true-cfg", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-mode", default="nearest", choices=("nearest", "bicubic"))
    ap.add_argument("--device", default="cuda")
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

    pipe = load_pipe(args.device)
    man = (out / "manifest.jsonl").open("w", encoding="utf-8")
    filt = Image.Resampling.BICUBIC if args.init_mode == "bicubic" else Image.Resampling.NEAREST

    with torch.no_grad():
        for j, (row, prompt) in enumerate(zip(rows, prompts)):
            mc = Image.fromarray(np.asarray(imgs[row]), "RGBA")
            bg = Image.new("RGB", mc.size, (255, 255, 255))
            bg.paste(mc.convert("RGB"), mask=mc.split()[3])
            init = bg.resize((args.hd_size, args.hd_size), filt)
            g = torch.Generator(device=args.device).manual_seed(args.seed + j)
            t0 = time.time()
            hd = pipe(image=[init], prompt=args.edit_prompt, negative_prompt=args.negative,
                      height=args.hd_size, width=args.hd_size,
                      num_inference_steps=args.steps,
                      true_cfg_scale=args.true_cfg, generator=g).images[0]
            el = time.time() - t0
            hd.save(out / "hd" / f"row{row}.png")
            hd.convert("RGB").resize((args.ref_size, args.ref_size),
                                     Image.Resampling.LANCZOS).save(out / "ref64" / f"row{row}.png")
            man.write(json.dumps({
                "row": row, "prompt": prompt, "mc_source": "real MC target",
                "hd_png": f"hd/row{row}.png", "ref_png": f"ref64/row{row}.png",
                "generator_model": "Qwen/Qwen-Image-Edit-2511",
                "generator_license": LICENSE,
                "edit_prompt": args.edit_prompt, "negative_prompt": args.negative,
                "steps": args.steps, "true_cfg": args.true_cfg,
                "init_mode": args.init_mode, "hd_size": args.hd_size,
                "seed": args.seed + j, "hd_seconds": round(el, 1),
            }) + "\n")
            man.flush()
            print(f"[{j}] row{row} :: {prompt[:55]}... hd {el:.1f}s", flush=True)
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
