"""Qwen-Image-Edit at native resolution with a stylised game-asset prompt.

Compares: 1024 realistic prompt (previous style) vs 1024 stylised game-asset
prompt vs 512 stylised, to separate resolution effects from prompt effects.

    python scripts/mc2hd_qwen_stylized.py --n 6 --out /tmp/opencode/qwen_styl
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
sys.path.insert(0, str(ROOT / "scripts"))

from try_antialias_filters import gauss, up_nearest  # noqa: E402

MODEL = "/home/iflab/models/Qwen-Image-Edit-2511"

REALISTIC = (
    "Redraw this as a realistic, high-quality 3D game asset texture: physically "
    "based materials, smooth continuous geometry, fine surface detail, clean "
    "anti-aliased edges, realistic lighting and shading, as a professional "
    "studio render. Keep the same object, the same orientation, the same "
    "proportions, the same silhouette and the same colors. Remove all "
    "pixelation, voxel blocks and stair-step edges.")

STYLIZED = (
    "Redraw this as a clean stylised game asset texture for a professional "
    "video game: a flat 2D texture that fills the entire frame, crisp "
    "well-defined shapes, simple painted shading, a small limited colour "
    "palette, subtle surface detail. Keep the same object, the same "
    "orientation, the same proportions, the same silhouette and the same "
    "colours. No photorealism, no photograph, no scene, no background, no "
    "environment, no text, no watermark.")

NEGATIVE = ("photorealistic, photograph, realistic photo, scene, background, "
            "environment, landscape, depth of field, dramatic lighting, "
            "3d render, voxel, voxels, pixelated, pixel art, minecraft, blocky, "
            "cubes, mosaic, grid, aliasing, flat untextured colors, text, "
            "watermark, logo, frame")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--true-cfg", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--blur", type=float, default=16.0)
    ap.add_argument("--prompt-parquet", default=str(
        ROOT / "data/build/mc_text2image32_wl/stage3_prompts.parquet"))
    ap.add_argument("--prompt-col", default="prompt_0")
    ap.add_argument("--build", default=str(ROOT / "data/build/mc_text2image32_wl"))
    ap.add_argument("--sizes", default="512,256,128",
                    help="comma list of square sizes to compare")
    ap.add_argument("--mode", default="styl", choices=("styl", "real"))
    ap.add_argument("--out", default="/tmp/opencode/qwen_styl")
    args = ap.parse_args()

    import pandas as pd
    from diffusers import QwenImageEditPlusPipeline

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
        bl = [i for i in s3["val"] if str(meta.at[i, "type"]) == "block"][:args.n // 2]
        it = [i for i in s3["val"] if str(meta.at[i, "type"]) == "item"][:args.n - args.n // 2]
        rows = bl + it
    gp = pd.read_parquet(args.prompt_parquet, columns=[args.prompt_col])

    pipe = QwenImageEditPlusPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    pipe.to("cuda:0")
    pipe.set_progress_bar_config(disable=True)

    runs = [(f"{sz}_{args.mode}", int(sz),
             STYLIZED if args.mode == "styl" else REALISTIC)
            for sz in args.sizes.split(",")]
    logs = []
    with torch.no_grad():
        for row in rows:
            item = str(meta.at[row, "type"]) == "item"
            subject = str(gp[args.prompt_col].iloc[row])
            mc = np.asarray(imgs[row])
            pil = Image.fromarray(mc, "RGBA")
            bg = Image.new("RGB", pil.size, (255, 255, 255))
            bg.paste(pil.convert("RGB"), mask=pil.split()[3])
            for tag, size, run_instr in runs:
                base = up_nearest(np.dstack([np.asarray(bg), np.full((32, 32), 255, np.uint8)]),
                                  size)
                v = gauss(base, args.blur * size / 512.0)
                img = Image.fromarray(v[..., :3].astype(np.uint8), "RGB")
                head = ("single item centered on a plain white background, " if item else "")
                pr = f"{head}{subject}. {run_instr}"
                g = torch.Generator(device="cuda:0").manual_seed(args.seed + row)
                t0 = time.time()
                hd = pipe(image=[img], prompt=pr, negative_prompt=NEGATIVE,
                          height=size, width=size, num_inference_steps=args.steps,
                          true_cfg_scale=args.true_cfg, generator=g).images[0]
                hd.save(out / f"row{row}_{tag}.png")
                el = round(time.time() - t0, 1)
                logs.append({"row": row, "tag": tag, "size": size, "sec": el,
                             "prompt": pr})
                print(f"row{row} {tag}: {el}s", flush=True)
    (out / "manifest.json").write_text(json.dumps(logs, indent=1))
    print("saved", out)


if __name__ == "__main__":
    main()
