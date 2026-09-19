import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import ModelConfig, load_yaml
from model.mc_flow_dit import MCFlowDiT
from data.embed_text import encode_texts
from .solver import sample, to_uint8


def load_model_from_checkpoint(ckpt_path, device="cuda", use_ema=False):
    sd = torch.load(ckpt_path, map_location="cpu")
    mcfg = ModelConfig.from_dict(sd["model_cfg"])
    model = MCFlowDiT(mcfg)
    model.load_state_dict(sd["model"])
    if use_ema and "ema" in sd:
        msd = model.state_dict()
        for k, v in sd["ema"]["shadow"].items():
            msd[k].copy_(v.to(dtype=msd[k].dtype))
        print("using EMA weights")
    model.to(device).eval()
    manifest = sd.get("conditioning", {})
    if manifest:
        print("conditioning:", {k: manifest[k] for k in sorted(manifest)})
    return model, mcfg, manifest


def encode_prompts(prompts, text_dim=768, max_tokens=64, model_name="", encoder_type="", instruction=""):
    if not encoder_type:
        encoder_type = "siglip2" if model_name else "hash"
    # Unconditional / placeholder conditioning is trained with a single null
    # token (see MmapImageTextDataset), so ignore max_tokens here. Real text
    # encoders keep their configured token count.
    tokens = 1 if encoder_type == "hash" else max_tokens
    return encode_texts(
        prompts,
        encoder_type=encoder_type,
        model_name=model_name,
        instruction=instruction,
        text_dim=text_dim,
        max_tokens=tokens,
    )


def sample_textures(model, prompts, seeds=None, steps=20, cfg=2.0, solver="heun", device="cuda", text_dim=768, max_tokens=64, text_encoder="", encoder_type="", instruction="", text_tower=None, premultiplied=False):
    cross = getattr(model, "text_injection", "joint") == "cross_attn"
    if cross:
        if text_tower is None:
            raise SystemExit("cross_attn model requires a text tower (--text-tower)")
        hidden, mask = text_tower.encode(prompts)
        text_all = hidden.to(device)
        mask_all = mask.to(device)
        uncond = torch.zeros_like(text_all)
        uncond_mask = torch.zeros_like(mask_all)
    else:
        embs = torch.from_numpy(
            encode_prompts(prompts, text_dim=text_dim, max_tokens=max_tokens, model_name=text_encoder, encoder_type=encoder_type, instruction=instruction)
        ).to(device)
        # Training drops the condition with the learned ``text_null`` parameter,
        # so CFG at inference must use the same null embedding (not zeros).
        text_null = getattr(model, "text_null", None)
        if text_null is not None:
            uncond = (text_null.detach().to(device=embs.device, dtype=embs.dtype)
                      .view(1, 1, -1).expand(embs.shape[0], 1, -1).contiguous())
        else:
            uncond = torch.zeros_like(embs)
    size = model.cfg.image_size
    results = []
    for i, p in enumerate(prompts):
        seed = seeds[i] if seeds is not None else 0
        g = torch.Generator(device=device).manual_seed(int(seed))
        z = torch.randn(1, model.cfg.in_channels, size, size, generator=g, device=device)
        if cross:
            x = sample(model, z, text_all[i:i + 1], steps=steps, cfg=cfg,
                       text_uncond=uncond[i:i + 1], solver=solver,
                       text_mask=mask_all[i:i + 1], text_uncond_mask=uncond_mask[i:i + 1])
        else:
            x = sample(model, z, embs[i:i + 1], steps=steps, cfg=cfg, text_uncond=uncond[i:i + 1], solver=solver)
        results.append(to_uint8(x[0], premultiplied=premultiplied))
    return results


def save_result(img, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = img.permute(1, 2, 0).cpu().numpy()
    Image.fromarray(arr).save(str(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt", nargs="+", required=True)
    ap.add_argument("--out", default="outputs/generated")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--solver", default="heun")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--text-encoder", default="")
    ap.add_argument("--encoder-type", default="", help="hash | siglip2 | qwen3vl")
    ap.add_argument("--instruction", default="")
    ap.add_argument("--use-ema", action="store_true")
    ap.add_argument("--text-tower", default="google/t5-v1_1-base",
                    help="HF token-level text encoder for cross_attn models")
    ap.add_argument("--text-max-length", type=int, default=512)
    ap.add_argument("--text-layers", default="",
                    help="comma list of hidden layers to concatenate, e.g. 9,18,27")
    args = ap.parse_args()

    model, mcfg, manifest = load_model_from_checkpoint(args.ckpt, args.device, use_ema=args.use_ema)
    premultiplied = bool(manifest.get("rgba_mode") == "premultiplied") if manifest else False
    text_tower = None
    if getattr(mcfg, "text_injection", "joint") == "cross_attn":
        from data.text_tower import get_text_encoder
        layers = [int(x) for x in args.text_layers.split(",") if x.strip()] if args.text_layers else None
        text_tower = get_text_encoder(args.text_tower, device=args.device,
                                      max_length=args.text_max_length, layers=layers)
    imgs = sample_textures(
        model,
        args.prompt,
        seeds=args.seeds,
        steps=args.steps,
        cfg=args.cfg,
        solver=args.solver,
        device=args.device,
        text_dim=mcfg.text_dim,
        max_tokens=mcfg.max_text_tokens,
        text_encoder=args.text_encoder,
        encoder_type=args.encoder_type,
        instruction=args.instruction,
        text_tower=text_tower,
        premultiplied=premultiplied,
    )
    for i, p in enumerate(args.prompt):
        out = Path(args.out) / f"{i:03d}_{'_'.join(p.split())[:40]}.png"
        save_result(imgs[i], out)
        print(f"saved {out}")


if __name__ == "__main__":
    main()