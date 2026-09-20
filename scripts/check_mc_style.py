"""Validate the corrected MC-ification recipe against real MC statistics.

Variants for each HD:
    hd32      : HD -> nearest 32                      (old, 1px blocks)
    t2        : HD -> nearest 16 -> nearest 32        (new, 2px blocks)
    t2q       : t2 + median-cut palette quantisation
    t3        : t2 -> SDEdit t0=0.3
    t3q       : t2q -> SDEdit t0=0.3

Metrics (mean over samples), with real MC as the reference target:
    blocks2   : fraction of 2x2 identical cells in the 32 grid (real MC = 1.0)
    colors    : number of unique RGB values
    edge      : mean absolute gradient
    sat       : mean saturation
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


def quantize(arr, colors):
    """arr: (32,32,3/4) uint8 -> median-cut quantised RGB with same shape."""
    im = Image.fromarray(arr[..., :3]).quantize(
        colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    return np.asarray(im.convert("RGB"))


def blocks2(arr):
    """Fraction of 2x2 cells that are uniform (real MC stored 32 from 16 = 1.0)."""
    a = arr[..., :3].astype(np.int16)
    d = (np.abs(a[0::2, 0::2] - a[0::2, 1::2]).sum(-1)
         + np.abs(a[0::2, 0::2] - a[1::2, 0::2]).sum(-1)
         + np.abs(a[0::2, 0::2] - a[1::2, 1::2]).sum(-1))
    return float((d == 0).mean())


def stats(arr):
    rgb = arr[..., :3].reshape(-1, 3)
    uq = len(np.unique(rgb, axis=0))
    mx = rgb.max(1).astype(float)
    mn = rgb.min(1).astype(float)
    sat = float(((mx - mn) / np.maximum(mx, 1)).mean())
    g = arr[..., :3].astype(float).mean(-1)
    edge = float(np.abs(np.diff(g, axis=0)).mean() + np.abs(np.diff(g, axis=1)).mean())
    return {"blocks2": round(blocks2(arr), 3), "colors": uq,
            "edge": round(edge, 1), "sat": round(sat, 3)}


def main():
    import pandas as pd

    n = 8
    man = [json.loads(l) for l in open(ROOT / "pairs/tierd_17k/manifest.jsonl")]
    bl = [m for m in man if m.get("asset_type") == "block"][:n // 2]
    it = [m for m in man if m.get("asset_type") == "item"][:n - n // 2]
    sel = bl + it
    prompts = [m["prompt"] for m in sel]

    model, _, m = load_model_from_checkpoint(
        str(ROOT / "checkpoints/stage_3_frozen_v2/best.pt"), "cuda", use_ema=False)
    pm = bool(m.get("rgba_mode") == "premultiplied")
    enc = get_text_encoder("/home/iflab/models/Qwen3-8B", device="cuda:1",
                           dtype="bfloat16", max_length=512, layers=[9, 18, 27])
    h, hm = enc.encode(prompts)
    nh, nm = torch.zeros_like(h[:1]), torch.zeros_like(hm[:1])

    build = ROOT / "data/build/mc_text2image32_wl"
    meta = pd.read_parquet(build / "metadata.parquet")
    imgs = np.memmap(build / "images.uint8.mmap", dtype=np.uint8, mode="r",
                     shape=(len(meta), 32, 32, 4))
    s3 = json.loads((build / "stage3_splits.json").read_text())
    real = [np.asarray(imgs[i]) for i in s3["val"][:n]]

    def sde(init_rgba, j, t0=0.3):
        x0 = torch.from_numpy(init_rgba).permute(2, 0, 1)[None].float().to("cuda") / 127.5 - 1.0
        if pm:
            x0 = premultiply_rgba_torch(x0)
        g = torch.Generator(device="cuda").manual_seed(j)
        z = torch.randn_like(x0)
        xt = (1 - t0) * x0 + t0 * z
        with torch.autocast("cuda", dtype=torch.bfloat16):
            xh = sdedit_euler(model, xt, t0, h[j:j + 1].to("cuda"), steps=20, cfg=2.5,
                              text_uncond=nh.to("cuda"), text_mask=hm[j:j + 1].to("cuda"),
                              text_uncond_mask=nm.to("cuda"))
        return to_uint8(xh[0], premultiplied=pm).permute(1, 2, 0).cpu().numpy()

    rows, agg = [], {k: [] for k in ("hd32", "t2", "t2q", "t3", "t3q")}
    with torch.no_grad():
        for j, e in enumerate(sel):
            hd = Image.open(e["hd_png"]).convert("RGB")
            hd32 = np.asarray(hd.resize((32, 32), Image.Resampling.NEAREST))
            small16 = np.asarray(hd.resize((16, 16), Image.Resampling.NEAREST))
            if e.get("asset_type") == "item":
                # keep white background transparent for items
                wk = np.asarray(hd.resize((16, 16), Image.Resampling.LANCZOS))
                alpha = np.where((wk >= 242).all(-1), 0, 255).astype(np.uint8)
            else:
                alpha = np.full((16, 16), 255, np.uint8)
            t2 = np.dstack([small16, alpha]).repeat(2, axis=0).repeat(2, axis=1)
            t2q = t2.copy()
            t2q[..., :3] = quantize(t2, 16)
            t3 = sde(t2, j, 0.3)
            t3q = sde(t2q, j, 0.3)
            variants = {"hd32": np.dstack([hd32, np.full((32, 32), 255, np.uint8)]),
                        "t2": t2, "t2q": t2q, "t3": t3, "t3q": t3q}
            for k, v in variants.items():
                vv = unpremultiply_rgba_np(v) if (pm and k in ("t3", "t3q")) else v
                agg[k].append(stats(vv))
            rows.append((e["prompt"], variants))

    print("real MC :", {k: round(float(np.mean([s[k] for s in [stats(r) for r in real]])), 3)
                        if k == "blocks2" else round(float(np.mean([stats(r)[k] for r in real])), 3)
                        for k in ("blocks2", "colors", "edge", "sat")})
    for k, v in agg.items():
        print(f"{k:6s}  :", {m: round(float(np.mean([x[m] for x in v])), 3)
                             for m in ("blocks2", "colors", "edge", "sat")})

    S, gap = 190, 8
    ncol = 6
    sheet = Image.new("RGB", (ncol * S + (ncol + 1) * gap, (len(rows) + 1) * (S + 22) + gap),
                      (25, 25, 25))
    d = ImageDraw.Draw(sheet)
    d.text((gap, 2), "HD | old HD->32 | t2 HD->16->32 | t2+quant16 | t3 SDEdit .3 | t3q SDEdit .3+quant",
           fill=(255, 255, 0))
    for r, (p, var) in enumerate(rows):
        y = 24 + r * (S + 22)
        d.text((gap, y), p[:70], fill=(150, 220, 255))
        arrs = [np.asarray(Image.open(sel[r]["hd_png"]).convert("RGB").resize((32, 32), Image.Resampling.LANCZOS)),
                var["hd32"], var["t2"], var["t2q"], var["t3"], var["t3q"]]
        for c, arr in enumerate(arrs):
            if arr.shape[-1] == 3:
                arr = np.dstack([arr, np.full(arr.shape[:2], 255, np.uint8)])
            arr = unpremultiply_rgba_np(arr) if (pm and c >= 4) else arr
            pil = Image.fromarray(arr[..., :4], "RGBA")
            bg = Image.new("RGB", pil.size, (255, 255, 255))
            bg.paste(pil.convert("RGB"), mask=pil.split()[3])
            sheet.paste(bg.resize((S, S), Image.Resampling.NEAREST), (gap + c * (S + gap), y + 18))
    y = 24 + len(rows) * (S + 22)
    d.text((gap, y), "real MC (target statistics)", fill=(255, 200, 200))
    for c, arr in enumerate(real):
        pil = Image.fromarray(arr, "RGBA")
        bg = Image.new("RGB", pil.size, (255, 255, 255))
        bg.paste(pil.convert("RGB"), mask=pil.split()[3])
        sheet.paste(bg.resize((S, S), Image.Resampling.NEAREST), (gap + c * (S + gap), y + 18))
    out = Path("/tmp/opencode/style_check.png")
    sheet.save(out)
    print("saved", out)


if __name__ == "__main__":
    main()
