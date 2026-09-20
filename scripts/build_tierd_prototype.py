"""Tier-D prototype: structured prompt -> FLUX.2-klein-4B HD -> nearest 32px ->
SDEdit with our MC model -> (prompt, MC) pair, reusing the HD prompt verbatim
as the MC annotation.

Provenance (model id, revision, seed, steps, guidance, prompt template,
license) is recorded per sample in manifest.jsonl.

    python scripts/build_tierd_prototype.py --prompts prompts.json --out pairs/tierd_proto
prompts.json: [{"prompt": "high-resolution game texture of ...", "asset_type": "block"}, ...]
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

from infer.sample import load_model_from_checkpoint  # noqa: E402
from infer.solver import to_uint8  # noqa: E402
from data.text_tower import get_text_encoder  # noqa: E402
from sample_sdedit import sdedit_euler  # noqa: E402

FLUX_MODEL = "/home/iflab/models/FLUX.2-klein-4B"
FLUX_LICENSE = "Apache-2.0 (black-forest-labs/FLUX.2-klein-4B)"

# Prompt wrapper: keep the layout faithful to a texture asset, no scene.
HD_SUFFIX = (", game texture asset filling the whole frame, flat front view, "
             "no background scene, no character, no text, no watermark")


def whitekey_alpha(hd_rgb, thresh=242):
    """Transparent background for item HDs: near-white pixels connected to the
    image border become transparent (flood fill, so white *inside* the item
    survives). Returns an RGBA array."""
    from scipy.ndimage import binary_propagation

    arr = np.asarray(hd_rgb.convert("RGB"), dtype=np.uint8)
    white = (arr >= thresh).all(axis=-1)
    seeds = np.zeros_like(white)
    seeds[0, :] = seeds[-1, :] = seeds[:, 0] = seeds[:, -1] = True
    bg = binary_propagation(seeds, mask=white)
    alpha = np.where(bg, 0, 255).astype(np.uint8)
    out = np.dstack([arr, alpha])
    return out


def load_flux(device):
    from diffusers import Flux2KleinPipeline

    pipe = Flux2KleinPipeline.from_pretrained(FLUX_MODEL, torch_dtype=torch.bfloat16)
    pipe.to(device)
    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass
    return pipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", default="pairs/tierd_proto")
    ap.add_argument("--mc-ckpt", default=str(ROOT / "checkpoints/stage_3_frozen_v2/best.pt"))
    ap.add_argument("--hd-size", type=int, default=512)
    ap.add_argument("--flux-steps", type=int, default=8)
    ap.add_argument("--guidance", type=float, default=3.5)
    ap.add_argument("--t0", type=float, default=0.5)
    ap.add_argument("--sde-steps", type=int, default=20)
    ap.add_argument("--cfg", type=float, default=2.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--flux-device", default="cuda:1")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "hd").mkdir(parents=True, exist_ok=True)
    (out / "mc").mkdir(parents=True, exist_ok=True)
    prompts = json.loads(Path(args.prompts).read_text())
    device, flux_device = args.device, args.flux_device

    flux = load_flux(flux_device)
    try:
        rev = json.load(open(f"{FLUX_MODEL}/model_index.json")).get("_diffusers_version", "?")
    except Exception:
        rev = "?"
    model, _, manifest = load_model_from_checkpoint(args.mc_ckpt, device, use_ema=False)
    premultiplied = bool(manifest.get("rgba_mode") == "premultiplied")
    enc = get_text_encoder("/home/iflab/models/Qwen3-8B", device=device,
                           dtype="bfloat16", max_length=512, layers=[9, 18, 27])

    man_path = out / "manifest.jsonl"
    man = man_path.open("a", encoding="utf-8")
    h_all, mask_all = enc.encode([p["prompt"] for p in prompts])
    null_h = torch.zeros_like(h_all[:1])
    null_m = torch.zeros_like(mask_all[:1])

    with torch.no_grad():
        for j, spec in enumerate(prompts):
            prompt, asset = spec["prompt"], spec.get("asset_type", "block")
            g = torch.Generator(device=flux_device).manual_seed(args.seed + j)
            t0 = time.time()
            hd = flux(image=None, prompt=prompt + HD_SUFFIX, height=args.hd_size,
                      width=args.hd_size, num_inference_steps=args.flux_steps,
                      guidance_scale=args.guidance, generator=g).images[0]
            hd_path = out / "hd" / f"{j:04d}.png"
            hd.save(hd_path)
            hd_t = time.time() - t0
            # nearest 32px init (bicubic-blurred input is OOD for the MC model)
            rgba = hd.convert("RGBA")
            if asset == "item":
                # Items need a transparent background: white-key the HD.
                rgba = Image.fromarray(whitekey_alpha(hd.convert("RGB")))
            init = np.asarray(rgba.resize((32, 32), Image.Resampling.NEAREST),
                              dtype=np.uint8)
            x0 = torch.from_numpy(init).permute(2, 0, 1)[None].float().to(device) / 127.5 - 1.0
            if premultiplied:
                from data.rgba import premultiply_rgba_torch

                x0 = premultiply_rgba_torch(x0)
            gz = torch.Generator(device=device).manual_seed(args.seed + j)
            z = torch.randn_like(x0)
            xt0 = (1.0 - args.t0) * x0 + args.t0 * z
            with torch.autocast("cuda", dtype=torch.bfloat16):
                xhat = sdedit_euler(
                    model, xt0, args.t0, h_all[j:j + 1].to(device), steps=args.sde_steps,
                    cfg=args.cfg, text_uncond=null_h.to(device),
                    text_mask=mask_all[j:j + 1].to(device), text_uncond_mask=null_m.to(device))
            mc = to_uint8(xhat[0], premultiplied=premultiplied).permute(1, 2, 0).cpu().numpy()
            mc_path = out / "mc" / f"{j:04d}.png"
            Image.fromarray(mc, "RGBA").save(mc_path)
            man.write(json.dumps({
                "id": j, "prompt": prompt, "asset_type": asset,
                "hd_png": str(hd_path), "mc_png": str(mc_path),
                "generator_model": "black-forest-labs/FLUX.2-klein-4B",
                "generator_license": FLUX_LICENSE,
                "diffusers_ref": rev,
                "flux_steps": args.flux_steps, "guidance": args.guidance,
                "hd_size": args.hd_size, "seed": args.seed + j,
                "mc_ckpt": args.mc_ckpt, "t0": args.t0,
                "sde_steps": args.sde_steps, "cfg": args.cfg,
                "hd_seconds": round(hd_t, 1),
            }) + "\n")
            man.flush()
            print(f"[{j}] {asset} :: {prompt[:70]}... hd {hd_t:.1f}s", flush=True)
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
