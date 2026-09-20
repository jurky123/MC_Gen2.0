"""MC -> HD with Qwen-Image-Edit (strong "realistic game asset" prompt).

Per-texture subject description + a strengthened realistic-asset instruction,
with a negative prompt suppressing voxel/pixelation (Qwen supports a real
negative prompt, unlike the step-distilled FLUX klein).

    python scripts/mc2hd_qwen_variants.py --n 6 --out /tmp/opencode/qwen_aa
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

from try_antialias_filters import bilateral, gauss, up_nearest  # noqa: E402

MODEL = "/home/iflab/models/Qwen-Image-Edit-2511"

EDIT_INSTRUCTION = (
    "Redraw this as a realistic, high-quality 3D game asset texture: physically "
    "based materials, smooth continuous geometry, fine surface detail, clean "
    "anti-aliased edges, realistic lighting and shading, as a professional "
    "studio render. Keep the same object, the same orientation, the same "
    "proportions, the same silhouette and the same colors. Remove all "
    "pixelation, voxel blocks and stair-step edges.")
NEGATIVE = ("pixel art, pixelated, voxel, voxels, blocky, cubes, minecraft, "
            "low resolution, jagged stair-step edges, mosaic, grid, aliasing, "
            "flat untextured colors, cartoon, cel shaded, background scene, "
            "text, watermark, logo")

PREPROC = {
    "gauss16": lambda a: gauss(a, 16),
    "gauss28": lambda a: gauss(a, 28),
    "bil+gauss12": lambda a: gauss(bilateral(a), 12),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--true-cfg", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variants", default="gauss16,gauss28,bil+gauss12")
    ap.add_argument("--prompt-parquet", default=str(
        ROOT / "data/build/mc_text2image32_wl/stage3_prompts.parquet"))
    ap.add_argument("--prompt-col", default="prompt_0")
    ap.add_argument("--build", default=str(ROOT / "data/build/mc_text2image32_wl"))
    ap.add_argument("--out", default="/tmp/opencode/qwen_aa")
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
    subjects = {i: str(gp[args.prompt_col].iloc[i]) for i in rows}

    pipe = QwenImageEditPlusPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    pipe.to("cuda:0")
    pipe.set_progress_bar_config(disable=True)

    results = {}
    with torch.no_grad():
        for row in rows:
            mc = np.asarray(imgs[row])
            pil = Image.fromarray(mc, "RGBA")
            bg = Image.new("RGB", pil.size, (255, 255, 255))
            bg.paste(pil.convert("RGB"), mask=pil.split()[3])
            base = up_nearest(np.dstack([np.asarray(bg), np.full((32, 32), 255, np.uint8)]),
                              args.size)
            pr = f"{subjects[row]}. {EDIT_INSTRUCTION}"
            per = {}
            for name in args.variants.split(","):
                v = PREPROC[name](base)
                img = Image.fromarray(v[..., :3].astype(np.uint8), "RGB")
                g = torch.Generator(device="cuda:0").manual_seed(args.seed + row)
                t0 = time.time()
                hd = pipe(image=[img], prompt=pr, negative_prompt=NEGATIVE,
                          height=args.size, width=args.size,
                          num_inference_steps=args.steps,
                          true_cfg_scale=args.true_cfg, generator=g).images[0]
                per[name] = {"input": v[..., :3], "hd": np.asarray(hd.convert("RGB")),
                             "sec": round(time.time() - t0, 1)}
                hd.save(out / f"row{row}_{name.replace('+', '_')}.png")
                print(f"row{row} {name}: {per[name]['sec']}s", flush=True)
            per["_prompt"] = pr
            results[row] = per
    (out / "manifest.json").write_text(json.dumps(
        {str(k): {"prompt": results[k]["_prompt"],
                  "variants": {n: {"sec": results[k][n]["sec"]}
                               for n in args.variants.split(",")}}
         for k in results}, indent=1))
    print("saved outputs to", out)


if __name__ == "__main__":
    main()
