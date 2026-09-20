"""Sample/preview a FLUX.2-klein-4B de-blocking LoRA.

For a set of real MC textures: upscale (NEAREST) to `res`, run the edit
pipeline with the optional LoRA adapter, and save a side-by-side sheet
(real MC | blocky input | no-LoRA baseline | LoRA output).

    python scripts/sample_lora_flux2.py --lora checkpoints/lora_flux2_deblock/last \
        --rows 156911,839889 --out /tmp/opencode/lora_preview
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FLUX_MODEL = "/home/iflab/models/FLUX.2-klein-4B"
EDIT_PROMPT = ("Turn this pixel-art game sprite into a smooth, highly detailed "
               "high-resolution game texture. Keep the same object, the same "
               "orientation, the same proportions, the same silhouette and the "
               "same colors. Round off the pixel stair-steps into smooth "
               "continuous shapes. Add realistic surface detail and clean "
               "anti-aliased edges.")
NEGATIVE = ("voxel, voxels, blocky, cube, cubic, minecraft, pixel art, "
            "pixelated, low resolution, jagged stair-step edges, mosaic, "
            "grid, aliasing, flat colors, background scene, text, watermark")


def blocky_input(mc_arr, res):
    pil = Image.fromarray(np.asarray(mc_arr, dtype=np.uint8), "RGBA")
    bg = Image.new("RGB", pil.size, (255, 255, 255))
    bg.paste(pil.convert("RGB"), mask=pil.split()[3])
    return bg.resize((res, res), Image.Resampling.NEAREST)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--lora", default="")
    ap.add_argument("--build", default=str(ROOT / "data/build/mc_text2image32_wl"))
    ap.add_argument("--prompt-parquet", default="")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="/tmp/opencode/lora_preview")
    args = ap.parse_args()

    import pandas as pd
    from diffusers import Flux2KleinPipeline

    build = Path(args.build)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [int(x) for x in args.rows.split(",") if x.strip()]
    meta = pd.read_parquet(build / "metadata.parquet")
    imgs = np.memmap(build / "images.uint8.mmap", dtype=np.uint8, mode="r",
                     shape=(len(meta), 32, 32, 4))
    pp = args.prompt_parquet or str(build / "stage3_prompts.parquet")
    gp = pd.read_parquet(pp, columns=["prompt_0"])

    pipe = Flux2KleinPipeline.from_pretrained(FLUX_MODEL, torch_dtype=torch.bfloat16)
    pipe.to(args.device)
    pipe.set_progress_bar_config(disable=True)
    baseline = pipe.transformer
    lora_pipe = None
    if args.lora:
        from peft import PeftModel

        lora_pipe = Flux2KleinPipeline.from_pretrained(FLUX_MODEL, torch_dtype=torch.bfloat16)
        lora_pipe.transformer = PeftModel.from_pretrained(lora_pipe.transformer, args.lora)
        lora_pipe.to(args.device)
        lora_pipe.transformer.eval()
        lora_pipe.set_progress_bar_config(disable=True)
    baseline.eval()

    def run(p, init, g):
        with torch.no_grad():
            return p(image=[init], prompt=EDIT_PROMPT, negative_prompt=NEGATIVE,
                     height=args.res, width=args.res, num_inference_steps=args.steps,
                     guidance_scale=4.0, generator=g).images[0]

    S, gap = 210, 8
    sheet = Image.new("RGB", (4 * S + 5 * gap, len(rows) * (S + 22) + gap), (25, 25, 25))
    from PIL import ImageDraw

    d = ImageDraw.Draw(sheet)
    d.text((gap, 2), f"real MC | blocky input | no-LoRA | LoRA({args.lora or '-'})",
           fill=(255, 255, 0))
    for r, row in enumerate(rows):
        y = 24 + r * (S + 22)
        prompt = str(gp["prompt_0"].iloc[row])
        d.text((gap, y), f"row{row} {prompt[:70]}", fill=(150, 220, 255))
        mc = Image.fromarray(np.asarray(imgs[row]), "RGBA")
        bg = Image.new("RGB", mc.size, (255, 255, 255))
        bg.paste(mc.convert("RGB"), mask=mc.split()[3])
        sheet.paste(bg.resize((S, S), Image.Resampling.NEAREST), (gap, y + 18))
        init = blocky_input(np.asarray(imgs[row]), args.res)
        sheet.paste(init.resize((S, S), Image.Resampling.NEAREST), (gap * 2 + S, y + 18))
        g = torch.Generator(device=args.device).manual_seed(args.seed + r)
        b = run(pipe, init, g)
        sheet.paste(b.resize((S, S), Image.Resampling.LANCZOS), (gap * 3 + S * 2, y + 18))
        if lora_pipe is not None:
            g2 = torch.Generator(device=args.device).manual_seed(args.seed + r)
            l = run(lora_pipe, init, g2)
            l.save(out / f"row{row}_lora.png")
            sheet.paste(l.resize((S, S), Image.Resampling.LANCZOS), (gap * 4 + S * 3, y + 18))
    sheet.save(out / "lora_vs_baseline.png")
    print(f"saved {out/'lora_vs_baseline.png'}")


if __name__ == "__main__":
    main()
