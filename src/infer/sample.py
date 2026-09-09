import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import ModelConfig, load_yaml
from model.mc_flow_dit import MCFlowDiT
from data.embed_text import hash_text_embed, siglip2_embed
from .solver import sample, to_uint8


def load_model_from_checkpoint(ckpt_path, device="cuda"):
    sd = torch.load(ckpt_path, map_location="cpu")
    mcfg = ModelConfig.from_dict(sd["model_cfg"])
    model = MCFlowDiT(mcfg)
    model.load_state_dict(sd["model"])
    model.to(device).eval()
    return model, mcfg


def encode_prompts(prompts, text_dim=768, max_tokens=64, model_name=""):
    if model_name:
        return siglip2_embed(prompts, model_name, max_tokens=max_tokens, token_dim=text_dim)
    return hash_text_embed(prompts, dim=text_dim, max_tokens=max_tokens)


def sample_textures(model, prompts, seeds=None, steps=20, cfg=2.0, solver="heun", device="cuda", text_dim=768, max_tokens=64, text_encoder=""):
    embs = torch.from_numpy(encode_prompts(prompts, text_dim=text_dim, max_tokens=max_tokens, model_name=text_encoder)).to(device)
    uncond = torch.zeros_like(embs)
    size = model.cfg.image_size
    results = []
    for i, p in enumerate(prompts):
        seed = seeds[i] if seeds is not None else 0
        g = torch.Generator(device=device).manual_seed(int(seed))
        z = torch.randn(1, 3, size, size, generator=g, device=device)
        text = embs[i : i + 1]
        u = uncond[i : i + 1]
        x = sample(model, z, text, steps=steps, cfg=cfg, text_uncond=u, solver=solver)
        results.append(to_uint8(x[0]))
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
    args = ap.parse_args()

    model, mcfg = load_model_from_checkpoint(args.ckpt, args.device)
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
    )
    for i, p in enumerate(args.prompt):
        out = Path(args.out) / f"{i:03d}_{'_'.join(p.split())[:40]}.png"
        save_result(imgs[i], out)
        print(f"saved {out}")


if __name__ == "__main__":
    main()