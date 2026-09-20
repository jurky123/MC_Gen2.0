"""FLUX edit with a realism-strengthened prompt, at several step counts.

The distilled klein needs the edit (image-conditioned) path; latent-SDEdit just
blurs. Here we push realism via the prompt and probe whether more inference
steps add detail.

    python scripts/mc2hd_flux_edit_grid.py --n 6 --steps-list 8,16,24
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

FLUX_MODEL = "/home/iflab/models/FLUX.2-klein-4B"

REALISM = (
    "Redraw this as a photorealistic high-end game asset: physically based "
    "materials (PBR), realistic lighting and shading, fine surface detail, "
    "crisp anti-aliased edges, professional studio render quality. Keep the "
    "same object, orientation, proportions, silhouette and colors. Remove all "
    "pixelation and voxel stair-step edges.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps-list", default="8,16,24")
    ap.add_argument("--blur", type=float, default=16.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt-parquet", default=str(
        ROOT / "data/build/mc_text2image32_wl/stage3_prompts.parquet"))
    ap.add_argument("--prompt-col", default="prompt_0")
    ap.add_argument("--build", default=str(ROOT / "data/build/mc_text2image32_wl"))
    ap.add_argument("--out", default="/tmp/opencode/flux_edit_grid")
    args = ap.parse_args()

    import pandas as pd
    from diffusers import Flux2KleinPipeline

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

    pipe = Flux2KleinPipeline.from_pretrained(FLUX_MODEL, torch_dtype=torch.bfloat16)
    pipe.to("cuda:1")
    pipe.set_progress_bar_config(disable=True)
    steps_list = [int(x) for x in args.steps_list.split(",")]

    logs = []
    with torch.no_grad():
        for row in rows:
            pr = f"{str(gp[args.prompt_col].iloc[row])}. {REALISM}"
            mc = np.asarray(imgs[row])
            pil = Image.fromarray(mc, "RGBA")
            bg = Image.new("RGB", pil.size, (255, 255, 255))
            bg.paste(pil.convert("RGB"), mask=pil.split()[3])
            base = up_nearest(np.dstack([np.asarray(bg), np.full((32, 32), 255, np.uint8)]),
                              args.size)
            v = gauss(base, args.blur).astype(np.uint8)
            init = Image.fromarray(v[..., :3])
            init.save(out / f"row{row}_init.png")
            for st in steps_list:
                g = torch.Generator(device="cuda:1").manual_seed(args.seed + row)
                t0 = time.time()
                hd = pipe(image=init, prompt=pr, height=args.size, width=args.size,
                          num_inference_steps=st, generator=g).images[0]
                hd.save(out / f"row{row}_steps{st}.png")
                logs.append({"row": row, "steps": st, "sec": round(time.time() - t0, 1),
                             "prompt": pr})
                print(f"row{row} steps={st}: {logs[-1]['sec']}s", flush=True)
    (out / "manifest.json").write_text(json.dumps(logs, indent=1))
    print("saved", out)


if __name__ == "__main__":
    main()
