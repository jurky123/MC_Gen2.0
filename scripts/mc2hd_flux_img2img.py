"""MC -> HD with FLUX.2-klein as latent img2img (SDEdit) at several noise levels.

Unlike the edit pipeline (image conditioning), here we VAE-encode the smoothed
MC, add noise at strength t0, and denoise with the FLUX flow model. This makes
the "how much freedom" knob explicit.

    python scripts/mc2hd_flux_img2img.py --n 6 --t0s 0.2,0.4,0.6,0.8
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
    "clean anti-aliased edges, as a professional studio render. Keep the same "
    "object, the same orientation, the same proportions, the same silhouette "
    "and the same colors. Remove all pixelation and voxel stair-step edges.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--t0s", default="0.2,0.4,0.6,0.8")
    ap.add_argument("--blur", type=float, default=16.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt-parquet", default=str(
        ROOT / "data/build/mc_text2image32_wl/stage3_prompts.parquet"))
    ap.add_argument("--prompt-col", default="prompt_0")
    ap.add_argument("--build", default=str(ROOT / "data/build/mc_text2image32_wl"))
    ap.add_argument("--out", default="/tmp/opencode/flux_i2i")
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
    dev = "cuda:1"
    vae_sf = pipe.vae_scale_factor

    def to_tensor(rgb):
        a = np.asarray(rgb, dtype=np.float32) / 127.5 - 1.0
        return torch.from_numpy(a).permute(2, 0, 1)[None].to(dev, dtype=torch.bfloat16)

    logs = []
    with torch.no_grad():
        for row in rows:
            subject = str(gp[args.prompt_col].iloc[row])
            pr = f"{subject}. {REALISM}"
            embeds, txt_ids = pipe.encode_prompt([pr], device=dev,
                                                 max_sequence_length=256,
                                                 text_encoder_out_layers=(9, 18, 27))
            mc = np.asarray(imgs[row])
            pil = Image.fromarray(mc, "RGBA")
            bg = Image.new("RGB", pil.size, (255, 255, 255))
            bg.paste(pil.convert("RGB"), mask=pil.split()[3])
            base = up_nearest(np.dstack([np.asarray(bg), np.full((32, 32), 255, np.uint8)]),
                              args.size)
            v = gauss(base, args.blur)
            init = v[..., :3].astype(np.uint8)
            Image.fromarray(init).save(out / f"row{row}_init.png")
            x0 = to_tensor(init)
            lat = pipe._encode_vae_image(image=x0, generator=None)
            ids = pipe._prepare_latent_ids(lat).to(dev)
            packed0 = pipe._pack_latents(lat)
            for t0 in [float(x) for x in args.t0s.split(",")]:
                g = torch.Generator(device=dev).manual_seed(args.seed + row)
                noise = torch.randn(lat.shape, generator=g, device=dev, dtype=lat.dtype)
                xt = (1 - t0) * lat + t0 * noise
                packed = pipe._pack_latents(xt)
                t_start = time.time()
                dt = t0 / args.steps
                for i in range(args.steps):
                    t = torch.full((1,), max(t0 - i * dt, 0.0), device=dev, dtype=lat.dtype)
                    vpred = pipe.transformer(hidden_states=packed.to(pipe.transformer.dtype),
                                             timestep=t, guidance=None,
                                             encoder_hidden_states=embeds.to(pipe.transformer.dtype),
                                             txt_ids=txt_ids, img_ids=ids,
                                             return_dict=False)[0]
                    packed = packed - dt * vpred
                lh = 2 * (args.size // (vae_sf * 2))
                lat_out = pipe._unpack_latents_with_ids(packed, ids, lh // 2, lh // 2)
                bn_mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(lat_out.device, lat_out.dtype)
                bn_std = torch.sqrt(pipe.vae.bn.running_var.view(1, -1, 1, 1)
                                    + pipe.vae.config.batch_norm_eps).to(lat_out.device, lat_out.dtype)
                lat_out = lat_out * bn_std + bn_mean
                lat_out = pipe._unpatchify_latents(lat_out)
                img = pipe.vae.decode(lat_out.to(pipe.vae.dtype), return_dict=False)[0]
                img = pipe.image_processor.postprocess(img, output_type="pil")[0]
                img.save(out / f"row{row}_t0_{t0}.png")
                logs.append({"row": row, "t0": t0, "sec": round(time.time() - t_start, 1)})
                print(f"row{row} t0={t0}: {logs[-1]['sec']}s", flush=True)
    (out / "manifest.json").write_text(json.dumps(logs, indent=1))
    print("saved", out)


if __name__ == "__main__":
    main()
