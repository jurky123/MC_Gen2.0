"""Human-review sheet for the HD->MC Stylizer.

For N validation rows:
    HD (FLUX 384) | real MC target | base t2i (text only) | Stylizer (text+ref)
plus a zoom row for the two generated columns so artefacts are visible.

    python scripts/review_stylizer.py --n 16 --out /tmp/opencode/styl_review
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import ModelConfig  # noqa: E402
from data.pair_dataset import MmapPairDataset  # noqa: E402
from data.rgba import unpremultiply_rgba_np  # noqa: E402
from data.text_tower import get_text_encoder  # noqa: E402
from infer.solver import to_uint8  # noqa: E402
from model.mc_flow_dit import MCFlowDiT  # noqa: E402


def load(ckpt, dev):
    sd = torch.load(ckpt, map_location="cpu")
    m = MCFlowDiT(ModelConfig.from_dict(sd["model_cfg"]))
    m.load_state_dict(sd["model"], strict=True)
    m.to(dev).eval()
    return m, sd.get("conditioning", {})


def sample_plain(model, z, text, tmask, nh, nm, steps=20, cfg=2.5):
    x = z
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((x.shape[0],), 1.0 - i * dt, device=x.device, dtype=x.dtype)
        vc = model(x, t, text, text_mask=tmask)
        vu = model(x, t, nh, text_mask=nm)
        x = x - dt * (vu + cfg * (vc - vu))
    return x.clamp(-1, 1)


def sample_ref(model, z, text, tmask, ref, nh, nm, steps=20, st=2.5, sr=2.5):
    x = z
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((x.shape[0],), 1.0 - i * dt, device=x.device, dtype=x.dtype)
        v_nn = model(x, t, nh, text_mask=nm, reference=None)
        v_tn = model(x, t, text, text_mask=tmask, reference=None)
        v_tt = model(x, t, text, text_mask=tmask, reference=ref)
        x = x - dt * (v_nn + st * (v_tn - v_nn) + sr * (v_tt - v_tn))
    return x.clamp(-1, 1)


def unp(arr, pm):
    return unpremultiply_rgba_np(arr) if pm else arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints/stylizer_phase1_v2/best.pt"))
    ap.add_argument("--base", default=str(ROOT / "checkpoints/stage_3_frozen_v2/best.pt"))
    ap.add_argument("--pairs", default=str(ROOT / "pairs/stylizer_v2"))
    ap.add_argument("--ref-size", type=int, default=64)
    ap.add_argument("--hd-dirs", default="pairs/mchd_stage3_a,pairs/mchd_stage3_b")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--st", type=float, default=2.5)
    ap.add_argument("--sr", type=float, default=2.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="/tmp/opencode/styl_review")
    args = ap.parse_args()

    import pandas as pd

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = args.device
    pairs = Path(args.pairs)
    ds = MmapPairDataset(
        ref_mmap=str(pairs / "ref.uint8.mmap"), target_mmap=str(pairs / "target.uint8.mmap"),
        metadata=str(pairs / "metadata.parquet"), splits=str(pairs / "splits.json"),
        split="val", ref_size=args.ref_size, target_size=32,
        rgba_mode="premultiplied")
    df = pd.read_parquet(pairs / "metadata.parquet")
    hd_map = {}
    for d in args.hd_dirs.split(","):
        d = Path(d)
        for p in d.glob("hd/row*.png"):
            hd_map[int(p.stem[3:])] = p

    idxs = list(range(min(args.n, len(ds))))
    rows = [int(df["row"].iloc[ds.index[i]]) for i in idxs]
    prompts = [str(df["prompt"].iloc[ds.index[i]]) for i in idxs]
    enc = get_text_encoder("/home/iflab/models/Qwen3-8B", device="cuda:1",
                           dtype="bfloat16", max_length=512, layers=[9, 18, 27])
    h, hm = enc.encode(prompts)
    nh, nm = torch.zeros_like(h).to(dev), torch.zeros_like(hm).to(dev)

    model, man = load(args.ckpt, dev)
    pm = bool(man.get("rgba_mode") == "premultiplied")
    base, bman = load(args.base, dev)
    bpm = bool(bman.get("rgba_mode") == "premultiplied")

    tiles = []
    with torch.no_grad():
        for k, i in enumerate(idxs):
            x_t, _p, aux = ds[i]
            ref = aux["reference"][None].to(dev)
            g = torch.Generator(device=dev).manual_seed(args.seed + k)
            z = torch.randn(1, 4, 32, 32, generator=g, device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                xb = sample_plain(base, z, h[k:k + 1].to(dev), hm[k:k + 1].to(dev),
                                  nh[k:k + 1], nm[k:k + 1], steps=args.steps)
                xs = sample_ref(model, z, h[k:k + 1].to(dev), hm[k:k + 1].to(dev), ref,
                                nh[k:k + 1], nm[k:k + 1], steps=args.steps,
                                st=args.st, sr=args.sr)
            tgt = unp(to_uint8(x_t, premultiplied=pm).permute(1, 2, 0).cpu().numpy(), pm)
            b = unp(to_uint8(xb[0], premultiplied=bpm).permute(1, 2, 0).cpu().numpy(), bpm)
            s = unp(to_uint8(xs[0], premultiplied=pm).permute(1, 2, 0).cpu().numpy(), pm)
            hd = (Image.open(hd_map[rows[k]]).convert("RGB") if rows[k] in hd_map else None)
            tiles.append((prompts[k], hd, tgt, b, s))
            print(f"[{k}] {prompts[k][:60]}", flush=True)

    def paste(sheet, arr_or_img, xy, size, nearest=False):
        if arr_or_img is None:
            return
        if isinstance(arr_or_img, np.ndarray):
            a = arr_or_img
            if a.shape[-1] == 3:
                a = np.dstack([a, np.full(a.shape[:2], 255, np.uint8)])
            pil = Image.fromarray(a[..., :4], "RGBA")
            bg = Image.new("RGB", pil.size, (255, 255, 255))
            bg.paste(pil.convert("RGB"), mask=pil.split()[3])
            im = bg
        else:
            im = arr_or_img
        sheet.paste(im.resize((size, size), Image.Resampling.NEAREST if nearest
                              else Image.Resampling.LANCZOS), xy)

    S, gap = 256, 8
    sheet = Image.new("RGB", (4 * S + 5 * gap, len(tiles) * (S + 26) + gap + 10), (24, 24, 24))
    d = ImageDraw.Draw(sheet)
    d.text((gap, 2), "HD (FLUX 384) | real MC target | base t2i (text only) | STYLIZER (text+ref)",
           fill=(255, 255, 0))
    for r, (p, hd, tgt, b, s) in enumerate(tiles):
        y = 24 + r * (S + 26)
        d.text((gap, y), f"{p[:96]}", fill=(150, 220, 255))
        paste(sheet, hd, (gap, y + 22), S)
        paste(sheet, tgt, (gap * 2 + S, y + 22), S, nearest=True)
        paste(sheet, b, (gap * 3 + S * 2, y + 22), S, nearest=True)
        paste(sheet, s, (gap * 4 + S * 3, y + 22), S, nearest=True)
    sheet.save(out / "review.png")
    print("saved", out / "review.png")


if __name__ == "__main__":
    main()
