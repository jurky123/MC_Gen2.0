"""Check: is the Tier-D MC block scale consistent with real MC?

Real MC is natively 16x16, stored as 32x32 with 2x2 colour blocks.
Tier-D built the MC as HD(512)->nearest 32, i.e. 16px blocks in the 32 grid.

This script SDEdits both init variants for the same HDs and shows them next to
real MC textures at the same 32 grid, so the block scale can be compared
directly.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from data.rgba import premultiply_rgba_torch, unpremultiply_rgba_np  # noqa: E402
from data.text_tower import get_text_encoder  # noqa: E402
from infer.sample import load_model_from_checkpoint  # noqa: E402
from infer.solver import to_uint8  # noqa: E402
from sample_sdedit import sdedit_euler  # noqa: E402


def build_init(hd, mode):
    if mode == "old16":
        small = hd.resize((32, 32), Image.Resampling.NEAREST)
    else:
        small = hd.resize((16, 16), Image.Resampling.NEAREST).resize(
            (32, 32), Image.Resampling.NEAREST)
    a = np.asarray(small.convert("RGB"))
    return np.dstack([a, np.full((32, 32), 255, np.uint8)])


def main():
    import pandas as pd

    man = [json.loads(l) for l in open(ROOT / "pairs/tierd_17k/manifest.jsonl")][:6]
    model, _, m = load_model_from_checkpoint(
        str(ROOT / "checkpoints/stage_3_frozen_v2/best.pt"), "cuda", use_ema=False)
    pm = bool(m.get("rgba_mode") == "premultiplied")
    enc = get_text_encoder("/home/iflab/models/Qwen3-8B", device="cuda:1",
                           dtype="bfloat16", max_length=512, layers=[9, 18, 27])
    prompts = [e["prompt"] for e in man]
    h, hm = enc.encode(prompts)
    nh, nm = torch.zeros_like(h[:1]), torch.zeros_like(hm[:1])

    # real MC references at 32px (native 16x16 upscaled) for scale comparison
    build = ROOT / "data/build/mc_text2image32_wl"
    meta = pd.read_parquet(build / "metadata.parquet")
    imgs = np.memmap(build / "images.uint8.mmap", dtype=np.uint8, mode="r",
                     shape=(len(meta), 32, 32, 4))
    s3 = json.loads((build / "stage3_splits.json").read_text())
    real_rows = s3["val"][:6]

    rows = []
    with torch.no_grad():
        for j, e in enumerate(man):
            hd = Image.open(e["hd_png"]).convert("RGB")
            outs = {}
            for mode in ("old16", "new2"):
                arr = build_init(hd, mode)
                x0 = torch.from_numpy(arr).permute(2, 0, 1)[None].float().to("cuda") / 127.5 - 1.0
                if pm:
                    x0 = premultiply_rgba_torch(x0)
                g = torch.Generator(device="cuda").manual_seed(j)
                z = torch.randn_like(x0)
                xt = (1 - 0.5) * x0 + 0.5 * z
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    xh = sdedit_euler(model, xt, 0.5, h[j:j + 1].to("cuda"), steps=20, cfg=2.5,
                                      text_uncond=nh.to("cuda"), text_mask=hm[j:j + 1].to("cuda"),
                                      text_uncond_mask=nm.to("cuda"))
                out = to_uint8(xh[0], premultiplied=pm).permute(1, 2, 0).cpu().numpy()
                outs[mode] = {"init": arr, "sde": out}
            rows.append((prompts[j], hd.resize((32, 32), Image.Resampling.NEAREST), outs))

    real = [np.asarray(imgs[r]) for r in real_rows]
    S, gap = 190, 8
    ncol = 5
    sheet = Image.new("RGB", (ncol * S + (ncol + 1) * gap,
                              (len(rows) + 1) * (S + 22) + 2 * gap), (25, 25, 25))
    d = ImageDraw.Draw(sheet)
    d.text((gap, 2), "HD | init old (16px blocks) | SDEdit(old) | init new (2px blocks) | SDEdit(new)",
           fill=(255, 255, 0))
    for r, (p, hd32, outs) in enumerate(rows):
        y = 24 + r * (S + 22)
        d.text((gap, y), p[:65], fill=(150, 220, 255))
        arrs = [np.asarray(hd32.convert("RGB")),
                outs["old16"]["init"], outs["old16"]["sde"],
                outs["new2"]["init"], outs["new2"]["sde"]]
        for c, arr in enumerate(arrs):
            if arr.shape[-1] == 3:
                arr = np.dstack([arr, np.full(arr.shape[:2], 255, np.uint8)])
            arr = np.asarray(Image.fromarray(arr, "RGBA") if arr.shape[-1] == 4 else arr)
            if pm and c in (2, 4):
                arr = unpremultiply_rgba_np(arr)
            pil = Image.fromarray(arr[..., :4], "RGBA")
            bg = Image.new("RGB", pil.size, (255, 255, 255))
            bg.paste(pil.convert("RGB"), mask=pil.split()[3])
            sheet.paste(bg.resize((S, S), Image.Resampling.NEAREST), (gap + c * (S + gap), y + 18))
    # a strip of real MC for scale
    y = 24 + len(rows) * (S + 22)
    d.text((gap, y), "real MC textures (native 16x16, stored 32 -> 2px blocks)", fill=(255, 200, 200))
    for c, arr in enumerate(real):
        pil = Image.fromarray(arr, "RGBA")
        bg = Image.new("RGB", pil.size, (255, 255, 255))
        bg.paste(pil.convert("RGB"), mask=pil.split()[3])
        sheet.paste(bg.resize((S, S), Image.Resampling.NEAREST), (gap + c * (S + gap), y + 18))
    out = Path("/tmp/opencode/res_check.png")
    sheet.save(out)
    print("saved", out)


if __name__ == "__main__":
    main()
