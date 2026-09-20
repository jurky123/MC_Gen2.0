"""LoRA fine-tune of FLUX.2-klein-4B for MC-style -> HD de-blocking.

Self-supervised pairs (no real HD ground truth needed):
    target  = an HD game texture (our Tier-D FLUX HDs, or any smooth HD image)
    input   = NEAREST-downsample(target) -> 32px -> NEAREST-upscale to `res`,
              optionally palette-quantised (median-cut, no dither) so the input
              looks like a restricted-palette MC texture.

The model is trained as an image-edit task with the same conditioning the
Flux2KleinPipeline uses at inference: reference latents are appended to the
target token sequence (time-offset >= 10) and the transformer predicts the
flow velocity for the target tokens only.

    python scripts/train_lora_flux2.py --manifest pairs/tierd_17k/manifest.jsonl \
        --out checkpoints/lora_flux2_deblock --res 512 --steps 2000
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FLUX_MODEL = "/home/iflab/models/FLUX.2-klein-4B"
LORA_TARGETS = ["to_q", "to_k", "to_v", "to_out.0", "to_qkv_mlp_proj",
                "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"]


def blockify(hd: Image.Image, res: int, quant_colors: int = 0) -> Image.Image:
    """HD -> NEAREST 32px (MC-like) -> NEAREST upscale to `res`, optional palette."""
    small = hd.convert("RGB").resize((32, 32), Image.Resampling.NEAREST)
    if quant_colors:
        small = small.quantize(colors=quant_colors,
                               method=Image.Quantize.MEDIANCUT,
                               dither=Image.Dither.NONE).convert("RGB")
    return small.resize((res, res), Image.Resampling.NEAREST)


def to_tensor(img: Image.Image):
    a = np.asarray(img.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(a).permute(2, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "pairs/tierd_17k/manifest.jsonl"))
    ap.add_argument("--root", default=str(ROOT / "pairs/tierd_17k"))
    ap.add_argument("--out", default=str(ROOT / "checkpoints/lora_flux2_deblock"))
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--quant-prob", type=float, default=0.5)
    ap.add_argument("--logit-normal", action="store_true", default=True)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from diffusers import Flux2KleinPipeline

    pipe = Flux2KleinPipeline.from_pretrained(FLUX_MODEL, torch_dtype=torch.bfloat16)
    pipe.to(args.device)
    pipe.set_progress_bar_config(disable=True)
    transformer, vae, text_encoder, tokenizer = (
        pipe.transformer, pipe.vae, pipe.text_encoder, pipe.tokenizer)
    for m in (vae, text_encoder):
        m.requires_grad_(False)
        m.eval()
    transformer.requires_grad_(False)

    from peft import LoraConfig, get_peft_model

    lcfg = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
                      init_lora_weights="gaussian", target_modules=LORA_TARGETS)
    transformer = get_peft_model(transformer, lcfg)
    transformer.enable_gradient_checkpointing()
    transformer.train()
    trainable = [p for p in transformer.parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in trainable)
    print(f"LoRA trainable {n_tr/1e6:.2f}M params (r={args.rank})")
    opt = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)

    root = Path(args.root)
    entries = [json.loads(l) for l in open(args.manifest) if l.strip()]
    rng.shuffle(entries)
    val = entries[-64:]
    entries = entries[:-64] if args.limit == 0 else entries[:args.limit]
    print(f"train pairs {len(entries)}, val {len(val)}")

    dtype = torch.bfloat16

    def resolve(p):
        p = Path(p)
        if p.is_absolute() and p.exists():
            return p
        for cand in (root / p, root / "hd" / p.name, ROOT / p):
            if cand.exists():
                return cand
        raise FileNotFoundError(p)

    def load_pair(e, quant_prob):
        hd = Image.open(resolve(e["hd_png"])).convert("RGB")
        qc = 0
        if rng.random() < quant_prob:
            qc = rng.choice([8, 12, 16, 24, 32])
        blk = blockify(hd, args.res, qc)
        tgt = hd.resize((args.res, args.res), Image.Resampling.LANCZOS)
        return blk, tgt, quant_prob

    def encode_images(pil_batch):
        x = torch.stack([to_tensor(p) for p in pil_batch]).to(args.device, dtype=dtype)
        with torch.no_grad():
            lat = pipe._encode_vae_image(image=x, generator=None)
        return lat  # (B, 128, H/16, W/16)

    step = 0
    t_start = time.time()
    while step < args.steps:
        for _ in range(args.grad_accum):
            batch = [entries[rng.randrange(len(entries))] for _ in range(args.batch)]
            pairs = [load_pair(e, args.quant_prob) for e in batch]
            blocks = [p[0] for p in pairs]
            targets = [p[1] for p in pairs]
            prompts = []
            for e in batch:
                p = str(e.get("prompt", "")).strip()
                if not p:
                    p = "a high quality game texture"
                if e.get("asset_type") == "item":
                    p = p + ", single item centered on a white background"
                prompts.append(p)

            with torch.no_grad():
                embeds, txt_ids = pipe.encode_prompt(
                    prompts, device=args.device, max_sequence_length=256,
                    text_encoder_out_layers=(9, 18, 27))
                ref_lat = encode_images(blocks)     # (B,128,h,w)
                x0_lat = encode_images(targets)

            B, C, Hl, Wl = x0_lat.shape
            latent_ids = pipe._prepare_latent_ids(x0_lat).to(args.device)
            # one reference per batch item: ids shape (B, H*W, 4) with T offset
            ref_ids = torch.cat(
                [pipe._prepare_image_ids([ref_lat[i:i + 1]]) for i in range(B)],
                dim=0).to(args.device)

            noise = torch.randn_like(x0_lat)
            if args.logit_normal:
                z = torch.randn(B, device=args.device) * 1.0
                t = torch.sigmoid(z)
            else:
                t = torch.rand(B, device=args.device)
            tb = t.view(B, 1, 1, 1)
            xt = (1.0 - tb) * x0_lat + tb * noise
            v_target = noise - x0_lat

            xt_p = pipe._pack_latents(xt)
            x0_p = pipe._pack_latents(x0_lat)
            ref_p = pipe._pack_latents(ref_lat)
            hidden = torch.cat([xt_p, ref_p], dim=1)
            img_ids = torch.cat([latent_ids, ref_ids], dim=1)

            v = transformer(
                hidden_states=hidden.to(dtype),
                timestep=t.to(dtype),
                guidance=None,
                encoder_hidden_states=embeds.to(dtype),
                txt_ids=txt_ids,
                img_ids=img_ids,
                return_dict=False,
            )[0]
            v = v[:, : x0_p.size(1)]
            loss = F.mse_loss(v.float(), v_target.reshape(B, x0_p.size(1), -1).float())
            (loss / args.grad_accum).backward()

        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % args.log_every == 0:
            el = time.time() - t_start
            print(f"step {step}/{args.steps} loss {loss.item():.4f} "
                  f"({el/step:.2f}s/step)", flush=True)
        if step % args.save_every == 0 or step == args.steps:
            transformer.save_pretrained(out / f"step{step}")
            (out / "last").mkdir(parents=True, exist_ok=True)
            transformer.save_pretrained(out / "last")
            print(f"saved LoRA -> {out}/step{step}", flush=True)
    transformer.save_pretrained(out / "last")
    print("done")


if __name__ == "__main__":
    main()
