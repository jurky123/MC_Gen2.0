"""v2 comparison sheet: 8 in-training (stage3 train) + 8 out-of-training
(global val, never trained) prompts; rows = real / stage2v2 gen / stage3v2 gen.
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from infer.sample import load_model_from_checkpoint  # noqa: E402
from infer.solver import sample, to_uint8  # noqa: E402
from data.text_tower import get_text_encoder  # noqa: E402

BUILD = ROOT / "data/build/mc_text2image32_wl"
CKPT2 = ROOT / "checkpoints/stage_2_grounded_k1_v2/best.pt"
CKPT3 = ROOT / "checkpoints/stage_3_frozen_v2/best.pt"
OUT = Path("/tmp/opencode/compare_s2_s3_v2.png")


def gen(model, enc, prompts, device, premultiplied, steps=24, cfg=2.5, seed=0):
    h, mask = enc.encode(prompts)
    null_h = torch.zeros_like(h[:1])
    null_m = torch.zeros_like(mask[:1])
    outs = []
    with torch.no_grad():
        for i in range(len(prompts)):
            g = torch.Generator(device=device).manual_seed(seed)
            z = torch.randn(1, model.cfg.in_channels, model.cfg.image_size,
                            model.cfg.image_size, generator=g, device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                x = sample(model, z, h[i:i + 1].to(device), steps=steps, cfg=cfg,
                           text_uncond=null_h.to(device), solver="heun",
                           text_mask=mask[i:i + 1].to(device),
                           text_uncond_mask=null_m.to(device))
            outs.append(to_uint8(x[0], premultiplied=premultiplied).permute(1, 2, 0).cpu().numpy())
    return outs


def main():
    import pandas as pd

    device = "cuda"
    tower_dev = "cuda:1" if torch.cuda.device_count() > 1 else "cuda"
    s3 = json.load(open(BUILD / "stage3_splits.json"))
    glob = json.load(open(BUILD / "splits.json"))
    meta = pd.read_parquet(BUILD / "metadata.parquet", columns=["type"])
    gp = pd.read_parquet(BUILD / "grounded_prompts.parquet", columns=["prompt_0"])
    s3p = pd.read_parquet(BUILD / "stage3_prompts.parquet", columns=["prompt_0"])
    N = len(meta)
    img = np.memmap(BUILD / "images.uint8.mmap", dtype=np.uint8, mode="r",
                    shape=(N, 32, 32, 4))

    rng = random.Random(7)
    a_blocks = [i for i in s3["train"] if meta["type"].values[i] == "block"]
    a_items = [i for i in s3["train"] if meta["type"].values[i] == "item"]
    picks_a = rng.sample(a_blocks, 4) + rng.sample(a_items, 4)
    prompts_a = [str(s3p["prompt_0"].iloc[i]) for i in picks_a]
    b_blocks = [i for i in glob["val"] if meta["type"].values[i] == "block"]
    b_items = [i for i in glob["val"] if meta["type"].values[i] == "item"]
    picks_b = rng.sample(b_blocks, 4) + rng.sample(b_items, 4)
    prompts_b = [str(gp["prompt_0"].iloc[i]) for i in picks_b]

    m2, _, man2 = load_model_from_checkpoint(str(CKPT2), device, use_ema=False)
    m3, _, man3 = load_model_from_checkpoint(str(CKPT3), device, use_ema=False)
    pm2 = bool(man2.get("rgba_mode") == "premultiplied")
    pm3 = bool(man3.get("rgba_mode") == "premultiplied")
    enc = get_text_encoder("/home/iflab/models/Qwen3-8B", device=tower_dev,
                           dtype="bfloat16", max_length=512, layers=[9, 18, 27])

    outs2_a = gen(m2, enc, prompts_a, device, pm2)
    outs3_a = gen(m3, enc, prompts_a, device, pm3)
    outs2_b = gen(m2, enc, prompts_b, device, pm2)
    outs3_b = gen(m3, enc, prompts_b, device, pm3)

    rows = [(picks_a, outs2_a, outs3_a, prompts_a), (picks_b, outs2_b, outs3_b, prompts_b)]
    S, gap, lab = 96, 4, 18
    W = 8 * S + 9 * gap
    RH = S + lab
    sheet = Image.new("RGB", (W, 6 * RH + 7 * gap), (30, 30, 30))
    labels = ["A-train REAL", "A-train STAGE2", "A-train STAGE3",
              "B-heldout REAL", "B-heldout STAGE2", "B-heldout STAGE3"]
    from PIL import ImageDraw

    d = ImageDraw.Draw(sheet)
    r = 0
    for picks, o2, o3, pr in rows:
        for arrs in (None, o2, o3):
            y0 = gap + r * (RH + gap)
            d.text((gap, y0), labels[r], fill=(255, 255, 0))
            for c in range(8):
                arr = np.asarray(img[picks[c]]) if arrs is None else arrs[c]
                pil = Image.fromarray(arr, "RGBA")
                bg = Image.new("RGB", pil.size, (255, 255, 255))
                bg.paste(pil.convert("RGB"), mask=pil.split()[3])
                sheet.paste(bg.resize((S, S), Image.Resampling.NEAREST),
                            (gap + c * (S + gap), y0 + lab))
            r += 1
    sheet.save(OUT)
    print(f"saved {OUT}")
    for pr in (prompts_a, prompts_b):
        for i, p in enumerate(pr):
            print(f"{'AB'[pr is prompts_b]}{i}: idx={[picks_a, picks_b][pr is prompts_b][i]} :: {p}")


if __name__ == "__main__":
    main()
